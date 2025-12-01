from typing import Dict, List, Optional, Tuple, Union

from functools import partial
import math
import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F
import torch_scatter

try:
    import spconv.pytorch as spconv
except:
    import spconv as spconv
from mmdet3d.models.layers.spconv import IS_SPCONV2_AVAILABLE    
if IS_SPCONV2_AVAILABLE:
    from spconv.pytorch import SparseConvTensor, SparseSequential
else:
    from mmcv.ops import SparseConvTensor, SparseSequential
    
from mmdet3d.registry import MODELS
from mmdet3d.models.layers import SparseBasicBlock, make_sparse_convmodule
from pcdet.ops.pointnet2.pointnet2_stack.pointnet2_utils import three_nn
from mamba_ssm import Block as MambaBlock
import time


@torch.inference_mode()
def get_window_coors_shift_v2(coords, sparse_shape, window_shape, shift=False):
    sparse_shape_z, sparse_shape_y, sparse_shape_x = sparse_shape
    win_shape_x, win_shape_y, win_shape_z = window_shape

    if shift:
        shift_x, shift_y, shift_z = win_shape_x // 2, win_shape_y // 2, win_shape_z // 2
    else:
        shift_x, shift_y, shift_z = 0, 0, 0  # win_shape_x, win_shape_y, win_shape_z

    max_num_win_x = int(np.ceil((sparse_shape_x / win_shape_x)) + 1)  # plus one here to meet the needs of shift.
    max_num_win_y = int(np.ceil((sparse_shape_y / win_shape_y)) + 1)  # plus one here to meet the needs of shift.
    max_num_win_z = int(np.ceil((sparse_shape_z / win_shape_z)) + 1)  # plus one here to meet the needs of shift.

    max_num_win_per_sample = max_num_win_x * max_num_win_y * max_num_win_z

    x = coords[:, 3] + shift_x
    y = coords[:, 2] + shift_y
    z = coords[:, 1] + shift_z

    win_coors_x = x // win_shape_x
    win_coors_y = y // win_shape_y
    win_coors_z = z // win_shape_z

    coors_in_win_x = x % win_shape_x
    coors_in_win_y = y % win_shape_y
    coors_in_win_z = z % win_shape_z

    batch_win_inds_x = coords[:, 0] * max_num_win_per_sample + win_coors_x * max_num_win_y * max_num_win_z + \
                       win_coors_y * max_num_win_z + win_coors_z
    batch_win_inds_y = coords[:, 0] * max_num_win_per_sample + win_coors_y * max_num_win_x * max_num_win_z + \
                       win_coors_x * max_num_win_z + win_coors_z

    coors_in_win = torch.stack([coors_in_win_z, coors_in_win_y, coors_in_win_x], dim=-1)

    return batch_win_inds_x, batch_win_inds_y, coors_in_win


def get_window_coors_shift_v1(coords, sparse_shape, window_shape):
    _, m, n = sparse_shape
    n2, m2, _ = window_shape

    n1 = int(np.ceil(n / n2) + 1)  # plus one here to meet the needs of shift.
    m1 = int(np.ceil(m / m2) + 1)  # plus one here to meet the needs of shift.

    x = coords[:, 3]
    y = coords[:, 2]

    x1 = x // n2
    y1 = y // m2
    x2 = x % n2
    y2 = y % m2

    return 2 * n2, 2 * m2, 2 * n1, 2 * m1, x1, y1, x2, y2


class FlattenedWindowMapping(nn.Module):
    def __init__(
            self,
            window_shape,
            group_size,
            shift,
            win_version='v2'
    ) -> None:
        super().__init__()
        self.window_shape = window_shape
        self.group_size = group_size
        self.win_version = win_version
        self.shift = shift

    def forward(self, coords: torch.Tensor, batch_size: int, sparse_shape: list):
        coords = coords.long()
        _, num_per_batch = torch.unique(coords[:, 0], return_counts=True)
        batch_start_indices = F.pad(torch.cumsum(num_per_batch, dim=0), (1, 0))
        num_per_batch_p = (
                torch.div(
                    batch_start_indices[1:] - batch_start_indices[:-1] + self.group_size - 1,
                    self.group_size,
                    rounding_mode="trunc",
                )
                * self.group_size
        )

        batch_start_indices_p = F.pad(torch.cumsum(num_per_batch_p, dim=0), (1, 0))
        flat2win = torch.arange(batch_start_indices_p[-1], device=coords.device)  # .to(coords.device)
        win2flat = torch.arange(batch_start_indices[-1], device=coords.device)  # .to(coords.device)
        key_padding_mask = torch.ones(batch_start_indices_p[-1], dtype=torch.bool, device=coords.device)

        for i in range(batch_size):
            if num_per_batch[i] != num_per_batch_p[i]:
                
                bias_index = batch_start_indices_p[i] - batch_start_indices[i]
                flat2win[
                    batch_start_indices_p[i + 1] - self.group_size + (num_per_batch[i] % self.group_size):
                    batch_start_indices_p[i + 1]
                    ] = flat2win[
                        batch_start_indices_p[i + 1]
                        - 2 * self.group_size
                        + (num_per_batch[i] % self.group_size): batch_start_indices_p[i + 1] - self.group_size
                        ] if (batch_start_indices_p[i + 1] - batch_start_indices_p[i]) - self.group_size != 0 else \
                        win2flat[batch_start_indices[i]: batch_start_indices[i + 1]].repeat(
                            (batch_start_indices_p[i + 1] - batch_start_indices_p[i]) // num_per_batch[i] + 1)[
                        : self.group_size - (num_per_batch[i] % self.group_size)] + bias_index

            win2flat[batch_start_indices[i]: batch_start_indices[i + 1]] += (
                    batch_start_indices_p[i] - batch_start_indices[i]
            )

            flat2win[batch_start_indices_p[i]: batch_start_indices_p[i + 1]] -= (
                    batch_start_indices_p[i] - batch_start_indices[i]
            )
            key_padding_mask[batch_start_indices_p[i]: batch_start_indices_p[i] + num_per_batch[i]] = 0

        mappings = {"flat2win": flat2win, "win2flat": win2flat, "key_padding_mask": key_padding_mask}

        get_win = self.win_version

        if get_win == 'v1':
            for shifted in [False]:
                (
                    n2,
                    m2,
                    n1,
                    m1,
                    x1,
                    y1,
                    x2,
                    y2,
                ) = get_window_coors_shift_v1(coords, sparse_shape, self.window_shape)
                vx = (n1 * y1 + (-1) ** y1 * x1) * n2 * m2 + (-1) ** y1 * (m2 * x2 + (-1) ** x2 * y2)
                vx += coords[:, 0] * sparse_shape[2] * sparse_shape[1] * sparse_shape[0]
                vy = (m1 * x1 + (-1) ** x1 * y1) * m2 * n2 + (-1) ** x1 * (n2 * y2 + (-1) ** y2 * x2)
                vy += coords[:, 0] * sparse_shape[2] * sparse_shape[1] * sparse_shape[0]
                _, mappings["x" + ("_shift" if shifted else "")] = torch.sort(vx)
                _, mappings["y" + ("_shift" if shifted else "")] = torch.sort(vy)

        elif get_win == 'v2':
            batch_win_inds_x, batch_win_inds_y, coors_in_win = get_window_coors_shift_v2(coords, sparse_shape,
                                                                                         self.window_shape, self.shift)
            vx = batch_win_inds_x * self.window_shape[0] * self.window_shape[1] * self.window_shape[2]
            vx += coors_in_win[..., 2] * self.window_shape[1] * self.window_shape[2] + coors_in_win[..., 1] * \
                  self.window_shape[2] + coors_in_win[..., 0]

            vy = batch_win_inds_y * self.window_shape[0] * self.window_shape[1] * self.window_shape[2]
            vy += coors_in_win[..., 1] * self.window_shape[0] * self.window_shape[2] + coors_in_win[..., 2] * \
                  self.window_shape[2] + coors_in_win[..., 0]

            _, mappings["x"] = torch.sort(vx)
            _, mappings["y"] = torch.sort(vy)

        elif get_win == 'v3':
            batch_win_inds_x, batch_win_inds_y, coors_in_win = get_window_coors_shift_v2(coords, sparse_shape,
                                                                                         self.window_shape)
            vx = batch_win_inds_x * self.window_shape[0] * self.window_shape[1] * self.window_shape[2]
            vx_xy = vx + coors_in_win[..., 2] * self.window_shape[1] * self.window_shape[2] + coors_in_win[..., 1] * \
                    self.window_shape[2] + coors_in_win[..., 0]
            vx_yx = vx + coors_in_win[..., 1] * self.window_shape[0] * self.window_shape[2] + coors_in_win[..., 2] * \
                    self.window_shape[2] + coors_in_win[..., 0]

            vy = batch_win_inds_y * self.window_shape[0] * self.window_shape[1] * self.window_shape[2]
            vy_xy = vy + coors_in_win[..., 2] * self.window_shape[1] * self.window_shape[2] + coors_in_win[..., 1] * \
                    self.window_shape[2] + coors_in_win[..., 0]
            vy_yx = vy + coors_in_win[..., 1] * self.window_shape[0] * self.window_shape[2] + coors_in_win[..., 2] * \
                    self.window_shape[2] + coors_in_win[..., 0]

            _, mappings["x_xy"] = torch.sort(vx_xy)
            _, mappings["y_xy"] = torch.sort(vy_xy)
            _, mappings["x_yx"] = torch.sort(vx_yx)
            _, mappings["y_yx"] = torch.sort(vy_yx)

        return mappings


class PatchMerging3D(nn.Module):
    def __init__(self, dim, out_dim=-1, down_scale=[2, 2, 2], norm_layer=nn.LayerNorm, diffusion=False, diff_scale=0.2, subm=True):
        super().__init__()
        self.dim = dim
        self.subm = subm
        
        if subm:
            self.sub_conv = spconv.SparseSequential(
                spconv.SubMConv3d(dim, dim, 3, bias=False, indice_key='subm'),
                nn.LayerNorm(dim),
                nn.GELU(),
            )
        else:
            self.linear = nn.Sequential(
                nn.Linear(dim, dim, bias=False),
                nn.LayerNorm(dim),
                nn.GELU(),
            )

        if out_dim == -1:
            self.norm = norm_layer(dim)
        else:
            self.norm = norm_layer(out_dim)

        self.sigmoid = nn.Sigmoid()
        self.down_scale = down_scale
        self.diffusion = diffusion
        self.diff_scale = diff_scale

        self.num_points = 6 #3

    def forward(self, x, coords_shift=1, diffusion_scale=4):
        assert diffusion_scale==4 or diffusion_scale==2

        if self.subm:
            x = self.sub_conv(x)
        else:
            x_feature = self.linear(x.features)
            x = x.replace_feature(x_feature)

        d, h, w = x.spatial_shape
        down_scale = self.down_scale

        if self.diffusion:
            x_feat_att = x.features.mean(-1)
            batch_size = x.indices[:, 0].max() + 1
            selected_diffusion_feats_list = [x.features.clone()]
            selected_diffusion_coords_list = [x.indices.clone()]
            for i in range(batch_size):
                mask = x.indices[:, 0] == i
                valid_num = mask.sum()
                K = int(valid_num * self.diff_scale)
                _, indices = torch.topk(x_feat_att[mask], K)

                selected_coords_copy = x.indices[mask][indices].clone()
                selected_coords_num = selected_coords_copy.shape[0]
                selected_coords_expand = selected_coords_copy.repeat(diffusion_scale, 1)
                selected_feats_expand = x.features[mask][indices].repeat(diffusion_scale, 1) * 0.0

                selected_coords_expand[selected_coords_num * 0:selected_coords_num * 1, 3:4] = (
                            selected_coords_copy[:, 3:4] - coords_shift).clamp(min=0, max=w - 1)
                selected_coords_expand[selected_coords_num * 0:selected_coords_num * 1, 2:3] = (
                            selected_coords_copy[:, 2:3] + coords_shift).clamp(min=0, max=h - 1)
                selected_coords_expand[selected_coords_num * 0:selected_coords_num * 1, 1:2] = (
                        selected_coords_copy[:, 1:2]).clamp(min=0, max=d - 1)

                selected_coords_expand[selected_coords_num:selected_coords_num * 2, 3:4] = (
                        selected_coords_copy[:, 3:4] + coords_shift).clamp(min=0, max=w - 1)
                selected_coords_expand[selected_coords_num:selected_coords_num * 2, 2:3] = (
                        selected_coords_copy[:, 2:3] + coords_shift).clamp(min=0, max=h - 1)
                selected_coords_expand[selected_coords_num:selected_coords_num * 2, 1:2] = (
                    selected_coords_copy[:, 1:2]).clamp(min=0, max=d - 1)

                if diffusion_scale==4:
#                         print('####diffusion_scale==4')
                    selected_coords_expand[selected_coords_num * 2:selected_coords_num * 3, 3:4] = (
                        selected_coords_copy[:, 3:4] - coords_shift).clamp(min=0, max=w - 1)
                    selected_coords_expand[selected_coords_num * 2:selected_coords_num * 3, 2:3] = (
                        selected_coords_copy[:, 2:3] - coords_shift).clamp(min=0, max=h - 1)
                    selected_coords_expand[selected_coords_num * 2:selected_coords_num * 3, 1:2] = (
                    selected_coords_copy[:, 1:2]).clamp(min=0, max=d - 1)

                    selected_coords_expand[selected_coords_num * 3:selected_coords_num * 4, 3:4] = (
                            selected_coords_copy[:, 3:4] + coords_shift).clamp(min=0, max=w - 1)
                    selected_coords_expand[selected_coords_num * 3:selected_coords_num * 4, 2:3] = (
                            selected_coords_copy[:, 2:3] - coords_shift).clamp(min=0, max=h - 1)
                    selected_coords_expand[selected_coords_num * 3:selected_coords_num * 4, 1:2] = (
                        selected_coords_copy[:, 1:2]).clamp(min=0, max=d - 1)

                selected_diffusion_coords_list.append(selected_coords_expand)
                selected_diffusion_feats_list.append(selected_feats_expand)

            coords = torch.cat(selected_diffusion_coords_list)
            final_diffusion_feats = torch.cat(selected_diffusion_feats_list)

        else:
            coords = x.indices.clone()
            final_diffusion_feats = x.features.clone()

        coords[:, 3:4] = coords[:, 3:4] // down_scale[0]
        coords[:, 2:3] = coords[:, 2:3] // down_scale[1]
        coords[:, 1:2] = coords[:, 1:2] // down_scale[2]

        scale_xyz = (x.spatial_shape[0] // down_scale[2]) * (x.spatial_shape[1] // down_scale[1]) * (
                x.spatial_shape[2] // down_scale[0])
        scale_yz = (x.spatial_shape[0] // down_scale[2]) * (x.spatial_shape[1] // down_scale[1])
        scale_z = (x.spatial_shape[0] // down_scale[2])

        merge_coords = coords[:, 0].int() * scale_xyz + coords[:, 3] * scale_yz + coords[:, 2] * scale_z + coords[:, 1]

        features_expand = final_diffusion_feats
        new_sparse_shape = [math.ceil(x.spatial_shape[i] / down_scale[2 - i]) for i in range(3)]
        unq_coords, unq_inv = torch.unique(merge_coords, return_inverse=True, return_counts=False, dim=0)

        x_merge = torch_scatter.scatter_add(features_expand, unq_inv, dim=0)

        unq_coords = unq_coords.int()
        voxel_coords = torch.stack((unq_coords // scale_xyz,
                                    (unq_coords % scale_xyz) // scale_yz,
                                    (unq_coords % scale_yz) // scale_z,
                                    unq_coords % scale_z), dim=1)
        voxel_coords = voxel_coords[:, [0, 3, 2, 1]]

        x_merge = self.norm(x_merge)

        x_merge = spconv.SparseConvTensor(
            features=x_merge,
            indices=voxel_coords.int(),
            spatial_shape=new_sparse_shape,
            batch_size=x.batch_size
        )
        return x_merge, unq_inv


class PatchExpanding3D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, up_x, unq_inv):
        # z, y, x
        n, c = x.features.shape

        x_copy = torch.gather(x.features, 0, unq_inv.unsqueeze(1).repeat(1, c))
        up_x = up_x.replace_feature(up_x.features + x_copy)
        return up_x


class LIONLayer(nn.Module):
    def __init__(self, dim, window_shape, group_size, direction, shift, mamba_cfg):
        super(LIONLayer, self).__init__()

        self.window_shape = window_shape
        self.group_size = group_size
        self.dim = dim
        self.direction = direction
        mamba_cfg['d_model'] = dim

        block_list = []
        for i in range(len(direction)):
            # operator_cfg['layer_id'] = i + layer_id
            # operator_cfg['n_layer'] = n_layer
            # operator_cfg['with_cp'] = layer_id >= 16
            # operator_cfg['with_cp'] = layer_id >= 0 ## all lion layer use checkpoint to save GPU memory!! (less 24G for training all models!!!)
            # print('### use part of checkpoint!!')
            block_list.append(MambaBlock(**mamba_cfg))

        self.blocks = nn.ModuleList(block_list)
        self.window_partition = FlattenedWindowMapping(self.window_shape, self.group_size, shift)

    def forward(self, x, pos_emb=None):
        
        mappings = self.window_partition(x.indices, x.batch_size, x.spatial_shape)

        for i, block in enumerate(self.blocks):
            indices = mappings[self.direction[i]]
            x_features = x.features[indices][mappings["flat2win"]]
            x_features = x_features.view(-1, self.group_size, x.features.shape[-1])
            if pos_emb is None:
                pos = torch.zeros_like(x_features)
            else:
                pos = pos_emb[indices][mappings["flat2win"]]
                pos = pos.view(-1, self.group_size, pos.shape[-1])
            x_features = block(x_features + pos)
            x.features[indices] = x_features.view(-1, x_features.shape[-1])[mappings["win2flat"]]
        
        return x

    
class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, input_channel, num_pos_feats):
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Linear(input_channel, num_pos_feats),
            nn.BatchNorm1d(num_pos_feats),
            nn.ReLU(inplace=True),
            nn.Linear(num_pos_feats, num_pos_feats))

    def forward(self, xyz):
        position_embedding = self.position_embedding_head(xyz)
        return position_embedding


class LIONBlock(nn.Module):
    def __init__(self, dim: int, depth: int, down_scales: list, window_shape, group_size, direction, mamba_cfg, shift=False, subm=True):
        super().__init__()

        if not isinstance(down_scales, list) and down_scales == 0:
            self.down_scales = None
        else:
            self.down_scales = down_scales

        self.encoder = nn.ModuleList()
        if self.down_scales is not None:
            self.downsample_list = nn.ModuleList()
        self.pos_emb_list = nn.ModuleList()

        norm_fn = partial(nn.LayerNorm)

        shift = [False, shift]
        for idx in range(depth):
            self.encoder.append(LIONLayer(dim, window_shape, group_size, direction, shift[idx], mamba_cfg))
            self.pos_emb_list.append(PositionEmbeddingLearned(input_channel=3, num_pos_feats=dim))
            if self.down_scales is not None:
                self.downsample_list.append(PatchMerging3D(dim, dim, down_scale=down_scales[idx], norm_layer=norm_fn, subm=subm))

        self.decoder = nn.ModuleList()
        self.decoder_norm = nn.ModuleList()
        if self.down_scales is not None:
            self.upsample_list = nn.ModuleList()
        for idx in range(depth):
            self.decoder.append(LIONLayer(dim, window_shape, group_size, direction, shift[idx], mamba_cfg))
            self.decoder_norm.append(norm_fn(dim))
            if self.down_scales is not None:
                self.upsample_list.append(PatchExpanding3D(dim))

    def forward(self, x):
        features = []
        index = []
        pos_emb_list = []

        for idx, enc in enumerate(self.encoder):
            pos_emb = self.get_pos_embed(spatial_shape=x.spatial_shape, coors=x.indices[:, 1:],
                                         embed_layer=self.pos_emb_list[idx])
            pos_emb_list.append(pos_emb)
            # x = replace_feature(x, pos_emb + x.features)  # x + pos_emb
            x = enc(x, pos_emb)
            features.append(x)
            if self.down_scales is not None:
                x, unq_inv = self.downsample_list[idx](x)
                index.append(unq_inv)
            else:
                index.append(None)

        i = 0
        for dec, norm, up_x, unq_inv in zip(self.decoder, self.decoder_norm, features[::-1], index[::-1]):
            x = dec(x, pos_emb_list[-(i+1)])
            if self.down_scales is not None:
                x = self.upsample_list[i](x, up_x, unq_inv)
                x = x.replace_feature(norm(x.features))
            else:
                x = x.replace_feature(norm(x.features + up_x.features))
            i = i + 1
        return x

    def get_pos_embed(self, spatial_shape, coors, embed_layer, normalize_pos=True):
        '''
        Args:
        coors_in_win: shape=[N, 3], order: z, y, x
        '''
        # [N,]
        window_shape = spatial_shape[::-1]  # spatial_shape:   win_z, win_y, win_x ---> win_x, win_y, win_z

        embed_layer = embed_layer
        if len(window_shape) == 2:
            ndim = 2
            win_x, win_y = window_shape
            win_z = 0
        # elif window_shape[-1] == 1:
        #     ndim = 2
        #     win_x, win_y = window_shape[:2]
        #     win_z = 0
        else:
            win_x, win_y, win_z = window_shape
            ndim = 3

        z, y, x = coors[:, 0] - win_z / 2, coors[:, 1] - win_y / 2, coors[:, 2] - win_x / 2

        if normalize_pos:
            x = x / win_x * 2 * 3.1415  # [-pi, pi]
            y = y / win_y * 2 * 3.1415  # [-pi, pi]
            z = z / win_z * 2 * 3.1415  # [-pi, pi]

        if ndim == 2:
            location = torch.stack((x, y), dim=-1)
        else:
            location = torch.stack((x, y, z), dim=-1)
        pos_embed = embed_layer(location)

        return pos_embed

    
@MODELS.register_module()
class UniMTFusionBackbone(nn.Module):
    
    def __init__(self, img_in_channels, lidar_in_channels, layer_dim, sparse_shape, pc_range,
                 encoder_channels = ((16, 16, 32), (32, 32, 64), (64, 64)),
                 encoder_paddings = ((0, 0, 1), (0, 0, 1), (0, 0)),
                 direction=['x', 'y'],
                 shift=True,
                 diffusion=True,
                 diff_scale=0.2,
                 patch_size=None,
                 mamba_cfg=dict(d_state=16, d_conv=4, expand=2, drop_path=0.2),
                 lidar2image=None,
                 image2lidar=None,
                 norm_cfg = dict(type='BN1d', eps=1e-3, momentum=0.01),
                 block_type = 'basicblock',
                 **kwargs):
        super().__init__()
        
        norm_fn = partial(nn.LayerNorm)
        self.sparse_shape = sparse_shape
        self.pc_range = pc_range
        self.encoder_channels = encoder_channels
        self.encoder_paddings = encoder_paddings
        patch_x, patch_y = torch.meshgrid(torch.arange(patch_size[0]), torch.arange(patch_size[1]), indexing='ij')
        patch_z = torch.zeros((patch_size[0] * patch_size[1], 1))
        self.patch_zyx = nn.Parameter(torch.cat([patch_z, patch_y.reshape(-1, 1), patch_x.reshape(-1, 1)], dim=-1), requires_grad=False)
        
        # image branch
        self.img_in = nn.Sequential(
            nn.Conv2d(img_in_channels, layer_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(layer_dim, eps=1e-3, momentum=0.01),
            nn.ReLU())
        self.img_out = nn.Sequential(
            nn.Conv2d(layer_dim, img_in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(img_in_channels, eps=1e-3, momentum=0.01),
            nn.ReLU())
        
        # lidar branch
        self.lidar_in = make_sparse_convmodule(lidar_in_channels, encoder_channels[0][0], 3,
                            norm_cfg=norm_cfg, indice_key='subm_in', conv_type='SubMConv3d', order=('conv', ))
        self.make_encoder_layers(make_sparse_convmodule, norm_cfg, encoder_channels[0][0], block_type=block_type)
        self.lidar_out = make_sparse_convmodule(encoder_channels[-1][-1], layer_dim, 3,
                            norm_cfg=norm_cfg, indice_key='subm_out', conv_type='SubMConv3d', order=('conv',))

        # lidar to image
        # self.linear_l2i = LIONBlock(layer_dim, lidar2image.depths[0], lidar2image.layer_down_scales[0],
        #                               lidar2image.window_shape[0], lidar2image.group_size[0], direction, mamba_cfg, shift=shift, subm=False)
        self.patch_down1 = PatchMerging3D(layer_dim, layer_dim, down_scale=[1, 1, 2],
                                    norm_layer=norm_fn, diffusion=diffusion, diff_scale=diff_scale)
        
        # image to lidar
        self.neighbor_pos_embed = PositionEmbeddingLearned(2, layer_dim)
        self.linear_i2l = LIONBlock(layer_dim, image2lidar.depths[0], image2lidar.layer_down_scales[0],
                                      image2lidar.window_shape[0], image2lidar.group_size[0], direction, mamba_cfg, shift=shift)
        self.patch_down2 = PatchMerging3D(layer_dim, layer_dim, down_scale=[1, 1, 2],
                                    norm_layer=norm_fn, diffusion=diffusion, diff_scale=diff_scale)

        self.linear_out = LIONLayer(layer_dim, [13, 13, 2], 256, direction, shift, mamba_cfg)


    def forward(self, voxel_features, voxel_coords, batch_size, img_feats, img_metas):
        
        # image branch
        img_feat = self.img_in(img_feats[1])
        BN = img_feat.shape[0]
        hw_shape = img_feat.shape[-2:]
        patch_features = img_feat.flatten(2).transpose(1, 2).contiguous()
        patch_features = patch_features.view(-1, patch_features.shape[-1])
        batch_idx = torch.arange(BN, device=patch_features.device).unsqueeze(1).repeat(1, hw_shape[0] * hw_shape[1]).view(-1, 1)
        patch_coords = torch.cat([batch_idx, self.patch_zyx.clone().to(batch_idx.device)[None, ::].repeat(BN, 1, 1).view(-1, 3)], dim=-1).long()
        
        # lidar branch
        x = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.lidar_in(x)
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
        x = self.lidar_out(x)
        
        # lidar to image
        # lidar2image_coors, hit_mask = map_lidar2image(x.indices, x.spatial_shape, self.pc_range, hw_shape, img_metas)
        # x_2d = spconv.SparseConvTensor(
        #     features=torch.cat([x.features[hit_mask].clone(), patch_features.clone()], dim=0),
        #     indices=torch.cat([lidar2image_coors[hit_mask], patch_coords], dim=0).int(),
        #     spatial_shape=[1, hw_shape[1], hw_shape[0]],
        #     batch_size=BN
        # )
        # x_2d = self.linear_l2i(x_2d)
        # x_temp = torch.zeros_like(x.features)
        # x_temp[hit_mask] = x_2d.features[:hit_mask.sum()]
        # x = x.replace_feature(x.features + x_temp)
        # patch_features = patch_features + x_2d.features[hit_mask.sum():]
        x, _ = self.patch_down1(x)
        
        # image to lidar
        image2lidar_coords, nearest_dist = map_image2lidar(patch_coords, x.spatial_shape, self.pc_range, hw_shape, img_metas)
        image2lidar_coords = torch.cat([patch_coords[:, :1].clone(), image2lidar_coords], dim=1)
        image2lidar_coords[:, 0] = image2lidar_coords[:, 0] // len(img_metas[0]['lidar2img'])
        neighbor_pos = self.neighbor_pos_embed(nearest_dist)
        x_3d = spconv.SparseConvTensor(
            features=torch.cat([x.features.clone(), patch_features.clone() + neighbor_pos], dim=0),
            indices=torch.cat([x.indices, image2lidar_coords], dim=0).int(),
            spatial_shape=x.spatial_shape,
            batch_size=batch_size
        )
        x_3d = self.linear_i2l(x_3d)
        x = x.replace_feature(x.features + x_3d.features[:x.features.shape[0]])
        patch_features = patch_features + x_3d.features[x.features.shape[0]:]
        
        x, _ = self.patch_down2(x)
        x = self.linear_out(x)
        pts_feats = x.dense()
        B, C, D, H, W = pts_feats.shape
        pts_feats = pts_feats.view(B, C * D, H, W)
        img_feat = patch_features.view(BN, hw_shape[0], hw_shape[1], -1).permute(0, 3, 1, 2).contiguous()
        img_feats[1] = self.img_out(img_feat)

        return pts_feats, img_feats
    
    
    def make_encoder_layers(
        self,
        make_block: nn.Module,
        norm_cfg: Dict,
        in_channels: int,
        block_type: Optional[str] = 'conv_module',
        conv_cfg: Optional[dict] = dict(type='SubMConv3d')
    ) -> int:
        """make encoder layers using sparse convs.

        Args:
            make_block (method): A bounded function to build blocks.
            norm_cfg (dict[str]): Config of normalization layer.
            in_channels (int): The number of encoder input channels.
            block_type (str, optional): Type of the block to use.
                Defaults to 'conv_module'.
            conv_cfg (dict, optional): Config of conv layer. Defaults to
                dict(type='SubMConv3d').

        Returns:
            int: The number of encoder output channels.
        """
        assert block_type in ['conv_module', 'basicblock']
        self.encoder_layers = SparseSequential()

        for i, blocks in enumerate(self.encoder_channels):
            blocks_list = []
            for j, out_channels in enumerate(tuple(blocks)):
                padding = tuple(self.encoder_paddings[i])[j]
                # each stage started with a spconv layer
                # except the first stage
                if i != 0 and j == 0 and block_type == 'conv_module':
                    blocks_list.append(
                        make_block(
                            in_channels,
                            out_channels,
                            3,
                            norm_cfg=norm_cfg,
                            stride=2,
                            padding=padding,
                            indice_key=f'spconv{i + 1}',
                            conv_type='SparseConv3d'))
                elif block_type == 'basicblock':
                    if j == len(blocks) - 1 and i != len(
                            self.encoder_channels) - 1:
                        blocks_list.append(
                            make_block(
                                in_channels,
                                out_channels,
                                3,
                                norm_cfg=norm_cfg,
                                stride=2,
                                padding=padding,
                                indice_key=f'spconv{i + 1}',
                                conv_type='SparseConv3d'))
                    else:
                        blocks_list.append(
                            SparseBasicBlock(
                                out_channels,
                                out_channels,
                                norm_cfg=norm_cfg,
                                conv_cfg=conv_cfg))
                else:
                    blocks_list.append(
                        make_block(
                            in_channels,
                            out_channels,
                            3,
                            norm_cfg=norm_cfg,
                            padding=padding,
                            indice_key=f'subm{i + 1}',
                            conv_type='SubMConv3d'))
                in_channels = out_channels
            stage_name = f'encoder_layer{i + 1}'
            stage_layers = SparseSequential(*blocks_list)
            self.encoder_layers.add_module(stage_name, stage_layers)
        return out_channels
    
    
def map_lidar2image(voxel_coords, sparse_shape, pc_range, hw_shape, img_metas):

    img_shape = img_metas[0]['pad_shape']
    lidar2image = np.stack([meta['lidar2img'] for meta in img_metas])
    lidar2image = torch.from_numpy(lidar2image).float().to(voxel_coords.device)
    batch_idx = voxel_coords[:, 0]
    num_view = lidar2image.shape[1]

    with torch.no_grad():
        sz, sy, sx = sparse_shape
        x1, y1, z1, x2, y2, z2 = pc_range
        points = voxel_coords[:, [3, 2, 1]].clone().float()
        points[..., 0] = ((points[..., 0]+0.5)/sx)*(x2-x1) + x1
        points[..., 1] = ((points[..., 1]+0.5)/sy)*(y2-y1) + y1
        points[..., 2] = ((points[..., 2]+0.5)/sz)*(z2-z1) + z1

        points = points.to(torch.float32)
        lidar2image = lidar2image.to(torch.float32)
        batch_size = (batch_idx[-1] + 1).int()
        batch_hit_points = []
        batch_hit_mask = []
        
        for b in range(batch_size):
            # map points from lidar to image space
            points_b = points[batch_idx == b]
            points_b = torch.cat([points_b, torch.ones_like(points_b[:, :1])], -1)
            points_b = points_b.unsqueeze(0).repeat(num_view, 1, 1).unsqueeze(-1)  # (num_view, grid_num, 4, 1)
            grid_num = points_b.shape[1]
            lidar2image_b = lidar2image[b].view(num_view, 1, 4, 4).repeat(1, grid_num, 1, 1)  # (num_view, grid_num, 4, 4)
            points_2d = torch.matmul(lidar2image_b, points_b).squeeze(-1)  # (num_view, grid_num, 4)
            eps = 1e-5
            map_mask = (points_2d[..., 2:3] > eps)
            points_2d = points_2d[..., 0:2] / torch.maximum(points_2d[..., 2:3], torch.ones_like(points_2d[..., 2:3]) * eps)
            points_2d[..., 0] /= img_shape[1]
            points_2d[..., 1] /= img_shape[0]

            # mask points out of range
            map_mask = (map_mask & (points_2d[..., 1:2] > 0.0) & (points_2d[..., 1:2] < 1.0)
                        & (points_2d[..., 0:1] < 1.0) & (points_2d[..., 0:1] > 0.0))
            map_mask = torch.nan_to_num(map_mask).squeeze(-1).T  # (grid_num, num_view)
            
            # get hit view id
            hit_mask = map_mask.any(dim=-1)
            map_mask[~hit_mask, 0] = True
            hit_view_ids = torch.nonzero(map_mask)

            # select first view if hit multi view
            hit_poins_id = hit_view_ids[:, 0]
            shift_hit_points_id = torch.roll(hit_poins_id, 1)
            shift_hit_points_id[0] = -1
            first_mask = (hit_poins_id - shift_hit_points_id) > 0
            unique_hit_view_ids = hit_view_ids[first_mask, 1:]

            # get coords in hit view
            points_2d = points_2d.permute(1, 0, 2)
            hit_points_2d = points_2d[range(grid_num), unique_hit_view_ids.squeeze()]

            # clamp value range and adjust to postive for set partition
            hit_points_2d = torch.clamp(hit_points_2d, 0, 1)
            hit_points = torch.cat([hit_points_2d, unique_hit_view_ids + b*num_view], -1)
            batch_hit_points.append(hit_points)
            batch_hit_mask.append(hit_mask)

        lidar2image_coords_xyz = torch.cat(batch_hit_points, dim=0)
        lidar2image_coords_xyz[:, 0] = lidar2image_coords_xyz[:, 0] * hw_shape[1]
        lidar2image_coords_xyz[:, 1] = lidar2image_coords_xyz[:, 1] * hw_shape[0]
        # lidar2image_coords_bzyx = torch.cat([voxel_coords[:, :1].clone(), lidar2image_coords_xyz[:, [2, 0, 1]]], dim=1)
        lidar2image_coords_bzyx = torch.cat([lidar2image_coords_xyz, torch.zeros_like(lidar2image_coords_xyz[:, :1])], -1)[:, [2, 3, 0, 1]]
        batch_hit_mask = torch.cat(batch_hit_mask, dim=0)

    return lidar2image_coords_bzyx, batch_hit_mask


def map_image2lidar(patch_coords, sparse_shape, pc_range, hw_shape, img_metas):

    img_shape = img_metas[0]['pad_shape']
    lidar2image = np.stack([meta['lidar2img'] for meta in img_metas])
    lidar2image = torch.from_numpy(lidar2image).float().to(patch_coords.device)
    batch_size = lidar2image.shape[0]
    num_view = lidar2image.shape[1]

    with torch.no_grad():

        sz, sy, sx = sparse_shape
        x1, y1, z1, x2, y2, z2 = pc_range
        coord_x = torch.linspace(0, sx-1, sx).view(1, -1, 1, 1).repeat(1, 1, sy, sz)
        coord_y = torch.linspace(0, sy-1, sy).view(1, 1, -1, 1).repeat(1, sx, 1, sz)
        coord_z = torch.linspace(0, sz-1, sz).view(1, 1, 1, -1).repeat(1, sx, sy, 1)
        coords = torch.stack((coord_x, coord_y, coord_z), -1).to(patch_coords.device).view(-1, 3)
        points = coords.clone().float()
        points[..., 0] = ((points[..., 0]+0.5)/sx)*(x2-x1) + x1
        points[..., 1] = ((points[..., 1]+0.5)/sy)*(y2-y1) + y1
        points[..., 2] = ((points[..., 2]+0.5)/sz)*(z2-z1) + z1

        points = points.to(torch.float32)
        lidar2image = lidar2image.to(torch.float32)

        # map points from lidar to (aug) image space
        points = torch.cat((points, torch.ones_like(points[:, :1])), -1)  # (grid_num, 4)
        points = points.unsqueeze(0).unsqueeze(0).repeat(batch_size, num_view, 1, 1).unsqueeze(-1)  # (batch_size, num_view, grid_num, 4, 1)
        grid_num = points.shape[2]
        lidar2image = lidar2image.view(batch_size, num_view, 1, 4, 4).repeat(1, 1, grid_num, 1, 1)  # (batch_size, num_view, grid_num, 4, 4)
        points_2d = torch.matmul(lidar2image, points).squeeze(-1)  # (batch_size, num_view, grid_num, 4)
        eps = 1e-5
        map_mask = (points_2d[..., 2:3] > eps)
        points_2d = points_2d[..., 0:2] / torch.maximum(points_2d[..., 2:3], torch.ones_like(points_2d[..., 2:3]) * eps)
        points_2d[..., 0] /= img_shape[1]
        points_2d[..., 1] /= img_shape[0]

        # mask points out of range
        map_mask = (map_mask & (points_2d[..., 1:2] > 0.0) & (points_2d[..., 1:2] < 1.0)
                    & (points_2d[..., 0:1] < 1.0) & (points_2d[..., 0:1] > 0.0))
        map_mask = torch.nan_to_num(map_mask).squeeze(-1)  # (batch_size, num_view, grid_num)
        
        # get mapping points in image space
        points_3d = points.squeeze(-1)  # (batch_size, num_view, grid_num, 4)
        mapped_points_2d = points_2d[map_mask]  # (N, 2)
        mapped_points_3d = points_3d[map_mask]  # (N, 4)
        mapped_view_cnts = map_mask.sum(-1).view(-1).int()  # (N)
        mapped_points = torch.cat([mapped_points_2d, torch.zeros_like(mapped_points_2d[:, :1])], dim=-1)  # (N, 3)
        mapped_coords_3d = mapped_points_3d[:, :3]  # (N, 3)

        # get image patch coords
        patch_coords_perimage = patch_coords[patch_coords[:, 0] == 0, 2:].clone().float()  # (H*W, 2)
        patch_coords_perimage[:, 0] = (patch_coords_perimage[:, 0] + 0.5) / hw_shape[1]
        patch_coords_perimage[:, 1] = (patch_coords_perimage[:, 1] + 0.5) / hw_shape[0]
        patch_points = patch_coords_perimage.unsqueeze(0).repeat(batch_size * num_view, 1, 1).view(-1, 2)
        patch_points = torch.cat([patch_points, torch.zeros_like(patch_points[:, :1])], dim=-1)  # (batch_size*num_view*H*W, 3)
        patch_view_cnts = (torch.ones_like(mapped_view_cnts) * (hw_shape[0] * hw_shape[1])).int()  # (N)

        # find the nearest 3 mapping points and keep the closest
        _, idx = three_nn(patch_points.to(torch.float32), patch_view_cnts, mapped_points.to(torch.float32), mapped_view_cnts)
        idx = idx[:, :1].repeat(1, 3).long()

        # take 3d coords of the nearest mapped point of each image patch as its 3d coords
        image2lidar_coords_xyz = torch.gather(mapped_coords_3d, 0, idx)

        # calculate distance between each image patch and the nearest mapping point in image space
        neighbor_2d = torch.gather(mapped_points, 0, idx)
        nearest_dist = (patch_points[:, :2]-neighbor_2d[:, :2]).abs()
        nearest_dist[:, 0] *= hw_shape[1]
        nearest_dist[:, 1] *= hw_shape[0]

        # 3d coords -> voxel grids
        image2lidar_coords_xyz[..., 0] = (image2lidar_coords_xyz[..., 0]-pc_range[0]) / (pc_range[3]-pc_range[0]) * sx - 0.5
        image2lidar_coords_xyz[..., 1] = (image2lidar_coords_xyz[..., 1]-pc_range[1]) / (pc_range[4]-pc_range[1]) * sy - 0.5
        image2lidar_coords_xyz[..., 2] = (image2lidar_coords_xyz[..., 2]-pc_range[2]) / (pc_range[5]-pc_range[2]) * sz - 0.5
        image2lidar_coords_xyz[..., 0] = torch.clamp(image2lidar_coords_xyz[..., 0], min=0, max=sx-1)
        image2lidar_coords_xyz[..., 1] = torch.clamp(image2lidar_coords_xyz[..., 1], min=0, max=sy-1)
        image2lidar_coords_zyx = image2lidar_coords_xyz[:, [2, 1, 0]]

        return image2lidar_coords_zyx, nearest_dist