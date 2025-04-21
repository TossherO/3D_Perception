import os
import os.path as osp
import time
import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn.bricks.transformer import MultiheadAttention
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttnFunction
from mmengine.model import BaseModule, ModuleList
from mmengine.model import xavier_init, uniform_init, constant_init


class DeformableAttention2MultiModality(BaseModule):
    
    def __init__(self, embed_dims=256, num_heads=8, num_points=4, dropout=0.1, im2col_step=64, batch_first=True):
        super(DeformableAttention2MultiModality, self).__init__()
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims ({embed_dims}) must be divisible by num_heads ({num_heads})')
        assert batch_first is True
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_points = num_points
        self.sampling_offsets = nn.Linear(embed_dims, num_heads * num_points * 3)
        self.cam_embedding = nn.Sequential(
            nn.Linear(12, embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims // 2, embed_dims),
            nn.ReLU(inplace=True),
            nn.LayerNorm(embed_dims)
        )
        self.pts_attention_weights = nn.Linear(embed_dims, num_heads * num_points)
        self.img_attention_weights = nn.Linear(embed_dims, num_heads * num_points)
        self.pts_proj = nn.Linear(embed_dims, embed_dims)
        self.img_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(2 * embed_dims, embed_dims)
        self.dropout = nn.Dropout(dropout)
        self.im2col_step = im2col_step
        self.init_weights()

    def init_weights(self):
        uniform_init(self.sampling_offsets, -2.0, 2.0)
        constant_init(self.pts_attention_weights, 0.0)
        constant_init(self.img_attention_weights, 0.0)
        xavier_init(self.pts_proj)
        xavier_init(self.img_proj)
        xavier_init(self.output_proj)

    def forward(self, query, query_pos, pts_feat, pts_pos, img_feat, img_pos, 
                reference_points, pc_range, img_metas):
        """Forward function for `DeformableAttention2MultiModality`.
        Args:
            query, query_pos (Tensor): [bs, num_query, embed_dims]
            pts_feat, pts_pos (Tensor): [bs, pts_l, pts_w, embed_dims]
            img_feat, img_pos (Tensor): [bs * num_cam, img_h, img_w, embed_dims]
            reference_points (Tensor): [bs, num_query, 3]
            pc_range (Tensor): [6]
            img_metas (dict): meta information, must contain 'lidar2img' 'pad_shape'
        Returns:
            Tensor: [bs, num_query, embed_dims]
        """
        bs, num_query, embed_dims = query.shape
        pts_l, pts_w = pts_feat.shape[1], pts_feat.shape[2]
        img_h, img_w = img_feat.shape[1], img_feat.shape[2]
        assert embed_dims == self.embed_dims
        num_cam = img_feat.shape[0] // bs
        assert num_cam == len(img_metas[0]['lidar2img'])
        assert pts_feat.shape[0] == img_feat.shape[0] // num_cam == bs
        assert query.shape == query_pos.shape
        assert pts_feat.shape == pts_pos.shape
        assert img_feat.shape == img_pos.shape

        # project pts_feat, img_feat
        pts_feat = self.pts_proj(pts_feat + pts_pos).view(bs, pts_l * pts_w, self.num_heads, -1)
        img_feat = self.img_proj(img_feat + img_pos).view(bs * num_cam, img_h * img_w, self.num_heads, -1)

        # get sampling offsets and attention
        identity = query
        query = query + query_pos
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_points, 3)
        pts_attention_weights = self.pts_attention_weights(query).view(
            bs, num_query, self.num_heads, 1, self.num_points).softmax(dim=-1)
        lidars2imgs = np.stack([meta['lidar2img'] for meta in img_metas])
        lidars2imgs = torch.from_numpy(lidars2imgs).float().to(query.device)    # [bs, num_cam, 4, 4]
        cam_embedding = self.cam_embedding(lidars2imgs[..., :3, :].flatten(-2)) # [bs, num_cam, embed_dims]
        query_cam = query.unsqueeze(1) + cam_embedding.unsqueeze(2)             # [bs, num_cam, num_query, embed_dims]
        img_attention_weights = self.img_attention_weights(query_cam).view(
            bs * num_cam, num_query, self.num_heads, 1, self.num_points).softmax(dim=-1)

        # get pts sampling points
        reference_points = reference_points * (pc_range[3:6] - pc_range[0:3]) + pc_range[0:3]
        sampling_points = reference_points.unsqueeze(2).unsqueeze(3) + sampling_offsets # [bs, num_query, num_heads, num_points, 3]
        sampling_points = torch.cat([sampling_points, torch.ones_like(sampling_points[..., :1])], dim=-1)
        pts_points = sampling_points[..., :2]
        pts_points[..., 0] = pts_points[..., 0] / (pc_range[3] - pc_range[0])
        pts_points[..., 1] = pts_points[..., 1] / (pc_range[4] - pc_range[1])
        pts_points = pts_points.view(bs, num_query, self.num_heads, 1, self.num_points, 2).contiguous()

        # get img sampling points
        img_points = torch.matmul(lidars2imgs[:, :, None, None, None], sampling_points[:, None, ..., None]).squeeze(-1)
        img_points = img_points[..., :2] / torch.clamp(img_points[..., 2:3], min=1e-5)
        img_points[..., 0] = img_points[..., 0] / img_metas[0]['pad_shape'][1]
        img_points[..., 1] = img_points[..., 1] / img_metas[0]['pad_shape'][0]
        # img_point_mask = (img_points[..., 0] >= 0) & (img_points[..., 0] <= 1) & (img_points[..., 1] >= 0) & (img_points[..., 1] <= 1) & (img_points[..., 2] > 0)
        img_points = img_points.view(bs*num_cam, num_query, self.num_heads, 1, self.num_points, 2).contiguous()
        
        # get pts, img features
        out_pts = MultiScaleDeformableAttnFunction.apply(
            pts_feat, torch.tensor([[pts_l, pts_w]]).to(query.device), torch.tensor([0]).to(query.device), pts_points, pts_attention_weights, self.im2col_step)
        out_img = MultiScaleDeformableAttnFunction.apply(
            img_feat, torch.tensor([[img_h, img_w]]).to(query.device), torch.tensor([0]).to(query.device), img_points, img_attention_weights, self.im2col_step)
        
        # get output
        out_img, _ = out_img.view(bs, num_cam, num_query, embed_dims).transpose(1, 2).max(dim=2)
        output = torch.cat([out_pts, out_img], dim=-1)
        output = self.output_proj(output)
        output = self.dropout(output) + identity
        return output


regular_attention = MultiheadAttention(embed_dims=256, num_heads=8, batch_first=True).cuda()
deformable_attention = DeformableAttention2MultiModality(embed_dims=256, num_heads=8).cuda()

lidar2img1 = np.array([[ 3.97734714e+02, -5.06729679e+02, -5.42572583e+01,
            2.74828461e+01],
          [ 2.46238289e+02, -3.78748864e+00, -5.36753266e+02,
           -5.95638454e+01],
          [ 9.91205872e-01, -1.70923051e-02, -1.31220294e-01,
           -3.00785658e-02],
          [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
            1.00000000e+00]])
lidar2img2 = np.array([[ 3.97710728e+02, -5.06737747e+02, -5.42431648e+01,
           -7.30179615e+01],
          [ 2.46215876e+02, -3.78133911e+00, -5.36727283e+02,
           -5.95612466e+01],
          [ 9.91180349e-01, -1.70659016e-02, -1.31184252e-01,
           -3.00764817e-02],
          [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
            1.00000000e+00]])
lidar2img = [lidar2img1, lidar2img2]
img_metas = [{'lidar2img': lidar2img, 'pad_shape': (640, 800)}]
pc_range = torch.tensor([-21.0, -21.0, -2.0, 21.0, 21.0, 6.0]).cuda()
query = torch.randn(1, 500, 256).cuda()
query_pos = torch.randn(1, 500, 256).cuda()
pts_feat = torch.randn(1, 70, 70, 256).cuda()
pts_pos = torch.randn(1, 70, 70, 256).cuda()
img_feat = torch.randn(2, 40, 50, 256).cuda()
img_pos = torch.randn(2, 40, 50, 256).cuda()
reference_points = torch.randint(0, 1, (1, 500, 3)).cuda()
key = torch.randn(1, 70*70 + 2*40*50, 256).cuda()
value = torch.randn(1, 70*70 + 2*40*50, 256).cuda()
key_pos = torch.randn(1, 70*70 + 2*40*50, 256).cuda()

# 运行前的显存占用
print('Memory Usage: %.3f MB' % (torch.cuda.memory_allocated() / 1024 / 1024))
output1 = regular_attention(query, key, value, query_pos=query_pos, key_pos=key_pos)
output2 = deformable_attention(query, query_pos, pts_feat, pts_pos, img_feat, img_pos, reference_points, pc_range, img_metas)
torch.cuda.empty_cache()

time1 = 0
time2 = 0
for i in range(200):
    query = torch.randn(1, 500, 256).cuda()
    query_pos = torch.randn(1, 500, 256).cuda()
    pts_feat = torch.randn(1, 70, 70, 256).cuda()
    pts_pos = torch.randn(1, 70, 70, 256).cuda()
    img_feat = torch.randn(2, 40, 50, 256).cuda()
    img_pos = torch.randn(2, 40, 50, 256).cuda()
    reference_points = torch.randint(0, 1, (1, 500, 3)).cuda()
    key = torch.randn(1, 70*70 + 2*40*50, 256).cuda()
    value = torch.randn(1, 70*70 + 2*40*50, 256).cuda()
    key_pos = torch.randn(1, 70*70 + 2*40*50, 256).cuda()
    
    now = time.time()
    output1 = regular_attention(query, key, value, query_pos=query_pos, key_pos=key_pos)
    time1 += time.time() - now
    
    torch.cuda.empty_cache()
    
    query = torch.randn(1, 500, 256).cuda()
    query_pos = torch.randn(1, 500, 256).cuda()
    pts_feat = torch.randn(1, 70, 70, 256).cuda()
    pts_pos = torch.randn(1, 70, 70, 256).cuda()
    img_feat = torch.randn(2, 40, 50, 256).cuda()
    img_pos = torch.randn(2, 40, 50, 256).cuda()
    reference_points = torch.randint(0, 1, (1, 500, 3)).cuda()
    key = torch.randn(1, 70*70 + 2*40*50, 256).cuda()
    value = torch.randn(1, 70*70 + 2*40*50, 256).cuda()
    key_pos = torch.randn(1, 70*70 + 2*40*50, 256).cuda()
    
    now = time.time()
    output2 = deformable_attention(query, query_pos, pts_feat, pts_pos, img_feat, img_pos, reference_points, pc_range, img_metas)
    time2 += time.time() - now
    
    torch.cuda.empty_cache()
    

print('Time1:', time1)
print('Time2:', time2)