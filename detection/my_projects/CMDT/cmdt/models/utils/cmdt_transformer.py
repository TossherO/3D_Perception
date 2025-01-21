# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------

import warnings
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from einops import rearrange
from mmengine.config import ConfigDict
from mmengine.model import BaseModule, ModuleList
from mmengine.model import xavier_init, uniform_init, constant_init
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import BaseTransformerLayer, TransformerLayerSequence
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttnFunction
from mmdet3d.registry import MODELS


# @MODELS.register_module()
# class CmdtTransformer(BaseModule):

#     def __init__(self, encoder=None, decoder=None, **kwargs):
#         super(CmdtTransformer, self).__init__(**kwargs)
#         if encoder is not None:
#             self.encoder = MODELS.build(encoder)
#         else:
#             self.encoder = None
#         self.decoder = MODELS.build(decoder)

#     def forward(self, query, query_pos, pts_feat, pts_pos, img_feat, img_pos, 
#                 reference_points, pc_range, img_metas, attn_masks=None):
#         """Forward function for `Transformer`.
#         Args:
#             query, query_pos (Tensor): [bs, num_query, embed_dims]
#             pts_feat, pts_pos (Tensor): [bs, pts_l, pts_w, embed_dims]
#             img_feat, img_pos (Tensor): [bs * num_cam, img_h, img_w, embed_dims]
#             reference_points (Tensor): [bs, num_query, 3]
#             pc_range (Tensor): [6]
#             img_metas (dict): meta information, must contain 'lidar2img' 'pad_shape'
#             attn_masks (Tensor): [num_query, num_query] or None
#         Returns:
#             out_dec (Tensor): [num_dec_layers, bs, num_query, embed_dims]
#         """
#         if self.encoder is not None:
#             pts_feat, img_feat = self.encoder(pts_feat, img_feat, pts_pos, img_pos, pc_range, img_metas)
#         out_dec = self.decoder(query, query_pos, pts_feat, pts_pos, img_feat, img_pos, 
#                                 reference_points, pc_range, img_metas, attn_masks)
#         return out_dec


@MODELS.register_module()
class CmdtTransformerDecoder(TransformerLayerSequence):

    def __init__(self,
                 *args,
                 post_norm_cfg=dict(type='LN'),
                 return_intermediate=False,
                 **kwargs):

        super(CmdtTransformerDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        if post_norm_cfg is not None:
            self.post_norm = build_norm_layer(post_norm_cfg, self.embed_dims)[1]
        else:
            self.post_norm = None

    def forward(self, query, query_pos, pts_feat, pts_pos, img_feat, img_pos, 
                reference_points, pc_range, img_metas, attn_masks=None, layer_idx=None):
        """Forward function for `TransformerDecoder`.
        Args:
            query, query_pos (Tensor): [bs, num_query, embed_dims]
            pts_feat, pts_pos (Tensor): [bs, pts_l, pts_w, embed_dims]
            img_feat, img_pos (Tensor): [bs * num_cam, img_h, img_w, embed_dims]
            reference_points (Tensor): [bs, num_query, 3]
            pc_range (Tensor): [6]
            img_metas (dict): meta information, must contain 'lidar2img' 'pad_shape'
            attn_masks (Tensor): [num_query, num_query] or None
        Returns:
            Tensor: If layer_idx is None, results with shape [1, num_query, bs, embed_dims] when 
            return_intermediate is `False`, otherwise it has shape [num_layers, num_query, bs, embed_dims].
            If layer_idx is not None, results with shape [bs, num_query, embed_dims].
        """
        if layer_idx is None:
            intermediate = []
            for layer in self.layers:
                query = layer(query, query_pos, pts_feat, pts_pos, img_feat, img_pos, 
                                reference_points, pc_range, img_metas, attn_masks)
                if self.return_intermediate:
                    if self.post_norm is not None:
                        intermediate.append(self.post_norm(query))
                    else:
                        intermediate.append(query)
            if self.return_intermediate:
                return torch.stack(intermediate)
            return query.unsqueeze(0)
        
        else:
            query = self.layers[layer_idx](query, query_pos, pts_feat, pts_pos, img_feat, img_pos,
                                            reference_points, pc_range, img_metas, attn_masks)
            if self.post_norm is not None:
                query = self.post_norm(query)
            return query


@MODELS.register_module()
class CmdtTransformerDecoderLayer(BaseModule):
    """Modified from base `TransformerLayer` for vision transformer.

    It can be built from `mmcv.ConfigDict` and support more flexible
    customization, for example, using any number of `FFN or LN ` and
    use different kinds of `attention` by specifying a list of `ConfigDict`
    named `attn_cfgs`. It is worth mentioning that it supports `prenorm`
    when you specifying `norm` as the first element of `operation_order`.
    More details about the `prenorm`: `On Layer Normalization in the
    Transformer Architecture <https://arxiv.org/abs/2002.04745>`_ .

    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | obj:`mmcv.ConfigDict` | None )):
            Configs for `self_attention` or `cross_attention` modules,
            The order of the configs in the list should be consistent with
            corresponding attentions in operation_order.
            If it is a dict, all of the attention modules in operation_order
            will be built with this config. Default: None.
        ffn_cfgs (list[`mmcv.ConfigDict`] | obj:`mmcv.ConfigDict` | None )):
            Configs for FFN, The order of the configs in the list should be
            consistent with corresponding ffn in operation_order.
            If it is a dict, all of the attention modules in operation_order
            will be built with this config.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Support `prenorm` when you specifying first element as `norm`.
            Default：None.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        batch_first (bool): Key, Query and Value are shape
            of (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
    """

    def __init__(self,
                 attn_cfgs=None,
                 ffn_cfgs=dict(
                     type='FFN',
                     embed_dims=256,
                     feedforward_channels=1024,
                     num_fcs=2,
                     ffn_drop=0.,
                     act_cfg=dict(type='ReLU', inplace=True),
                 ),
                 operation_order=None,
                 norm_cfg=dict(type='LN'),
                 init_cfg=None,
                 batch_first=False,
                 with_cp=False,
                 **kwargs):

        deprecated_args = dict(
            feedforward_channels='feedforward_channels',
            ffn_dropout='ffn_drop',
            ffn_num_fcs='num_fcs')
        for ori_name, new_name in deprecated_args.items():
            if ori_name in kwargs:
                warnings.warn(
                    f'The arguments `{ori_name}` in BaseTransformerLayer '
                    f'has been deprecated, now you should set `{new_name}` '
                    f'and other FFN related arguments '
                    f'to a dict named `ffn_cfgs`. ', DeprecationWarning)
                ffn_cfgs[new_name] = kwargs[ori_name]

        super().__init__(init_cfg)

        self.batch_first = batch_first

        assert set(operation_order) & {
            'self_attn', 'norm', 'ffn', 'cross_attn'} == \
            set(operation_order), f'The operation_order of' \
            f' {self.__class__.__name__} should ' \
            f'contains all four operation type ' \
            f"{['self_attn', 'norm', 'ffn', 'cross_attn']}"

        num_attn = operation_order.count('self_attn') + operation_order.count(
            'cross_attn')
        if isinstance(attn_cfgs, dict):
            attn_cfgs = [copy.deepcopy(attn_cfgs) for _ in range(num_attn)]
        else:
            assert num_attn == len(attn_cfgs), f'The length ' \
                f'of attn_cfg {num_attn} is ' \
                f'not consistent with the number of attention' \
                f'in operation_order {operation_order}.'

        self.num_attn = num_attn
        self.operation_order = operation_order
        self.norm_cfg = norm_cfg
        self.pre_norm = operation_order[0] == 'norm'
        self.attentions = ModuleList()

        index = 0
        for operation_name in operation_order:
            if operation_name in ['self_attn', 'cross_attn']:
                if 'batch_first' in attn_cfgs[index]:
                    assert self.batch_first == attn_cfgs[index]['batch_first']
                else:
                    attn_cfgs[index]['batch_first'] = self.batch_first
                attention = MODELS.build(attn_cfgs[index])
                # Some custom attentions used as `self_attn`
                # or `cross_attn` can have different behavior.
                attention.operation_name = operation_name
                self.attentions.append(attention)
                index += 1

        self.embed_dims = self.attentions[0].embed_dims

        self.ffns = ModuleList()
        num_ffns = operation_order.count('ffn')
        if isinstance(ffn_cfgs, dict):
            ffn_cfgs = ConfigDict(ffn_cfgs)
        if isinstance(ffn_cfgs, dict):
            ffn_cfgs = [copy.deepcopy(ffn_cfgs) for _ in range(num_ffns)]
        assert len(ffn_cfgs) == num_ffns
        for ffn_index in range(num_ffns):
            if 'embed_dims' not in ffn_cfgs[ffn_index]:
                ffn_cfgs[ffn_index]['embed_dims'] = self.embed_dims
            else:
                assert ffn_cfgs[ffn_index]['embed_dims'] == self.embed_dims
            self.ffns.append(MODELS.build(ffn_cfgs[ffn_index]))

        self.norms = ModuleList()
        num_norms = operation_order.count('norm')
        for _ in range(num_norms):
            self.norms.append(build_norm_layer(norm_cfg, self.embed_dims)[1])
        self.use_checkpoint = with_cp
    
    def _forward(self, query, query_pos, pts_feat, pts_pos, img_feat, img_pos, 
                    reference_points, pc_range, img_metas, attn_masks):
        """Forward function for `TransformerCoder`.
        Returns:
            Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """
        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                        f'attn_masks {len(attn_masks)} must be equal ' \
                        f'to the number of attention in ' \
                        f'operation_order {self.num_attn}'

        for layer in self.operation_order:
            if layer == 'self_attn':
                temp_key = temp_value = query
                query = self.attentions[attn_index](
                    query,
                    temp_key,
                    temp_value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=query_pos,
                    attn_mask=attn_masks[attn_index])
                attn_index += 1
                identity = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == 'cross_attn':
                query = self.attentions[attn_index](
                    query, query_pos, pts_feat, pts_pos, img_feat, img_pos, 
                    reference_points, pc_range, img_metas)
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1

        return query

    def forward(self, query, query_pos, pts_feat, pts_pos, img_feat, img_pos, 
                reference_points, pc_range, img_metas, attn_masks):
        """Forward function for `TransformerCoder`.
        Returns:
            Tensor: forwarded results with shape [bs, num_query, embed_dims].
        """

        if self.use_checkpoint and self.training:
            x = cp.checkpoint(self._forward, query, query_pos, pts_feat, pts_pos, img_feat, img_pos, 
                                reference_points, pc_range, img_metas, attn_masks)
        else:
            x = self._forward(query, query_pos, pts_feat, pts_pos, img_feat, img_pos, 
                                reference_points, pc_range, img_metas, attn_masks)
        return x
    

@MODELS.register_module()
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