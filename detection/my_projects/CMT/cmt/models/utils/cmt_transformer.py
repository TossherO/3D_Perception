# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------

import torch
from einops import rearrange
from mmengine.model import BaseModule
from mmengine.model import xavier_init
from mmdet3d.registry import MODELS


@MODELS.register_module()
class CmtTransformer(BaseModule):
    """Implements the DETR transformer.
    Following the official DETR implementation, this module copy-paste
    from torch.nn.Transformer with modifications:
        * positional encodings are passed in MultiheadAttention
        * extra LN at the end of encoder is removed
        * decoder returns a stack of activations from all decoding layers
    See `paper: End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.
    Args:
        encoder (`mmcv.ConfigDict` | Dict): Config of
            TransformerEncoder. Defaults to None.
        decoder ((`mmcv.ConfigDict` | Dict)): Config of
            TransformerDecoder. Defaults to None
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Defaults to None.
    """

    def __init__(self, encoder=None, decoder=None, init_cfg=None, cross=False):
        super(CmtTransformer, self).__init__(init_cfg=init_cfg)
        if encoder is not None:
            self.encoder = MODELS.build(encoder)
        else:
            self.encoder = None
        self.decoder = MODELS.build(decoder)
        self.embed_dims = self.decoder.embed_dims
        self.cross = cross

    def init_weights(self):
        # follow the official DETR to init parameters
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                xavier_init(m, distribution='uniform')
        self._is_init = True

    def forward(self, x, x_img, query_embed, bev_pos_embed, rv_pos_embed, attn_masks=None, reg_branch=None):
        """Forward function for `Transformer`.
        Args:
            x (Tensor): Input query with shape [bs, c, h, w] where
                c = embed_dims.
            mask (Tensor): The key_padding_mask used for encoder and decoder,
                with shape [bs, h, w].
            query_embed (Tensor): The query embedding for decoder, with shape
                [num_query, c].
            pos_embed (Tensor): The positional encoding for encoder and
                decoder, with the same shape as `x`.
        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.
                - out_dec: Output from decoder. If return_intermediate_dec \
                      is True output has shape [num_dec_layers, bs,
                      num_query, embed_dims], else has shape [1, bs, \
                      num_query, embed_dims].
                - memory: Output results from encoder, with shape \
                      [bs, embed_dims, h, w].
        """
        bs, c, h, w = x.shape
        bev_memory = rearrange(x, "bs c h w -> (h w) bs c") # [bs, n, c, h, w] -> [n*h*w, bs, c]
        rv_memory = rearrange(x_img, "(bs v) c h w -> (v h w) bs c", bs=bs)
        bev_pos_embed = bev_pos_embed.unsqueeze(1).repeat(1, bs, 1) # [bs, n, c, h, w] -> [n*h*w, bs, c]
        rv_pos_embed = rearrange(rv_pos_embed, "(bs v) h w c -> (v h w) bs c", bs=bs)
        
        memory, pos_embed = torch.cat([bev_memory, rv_memory], dim=0), torch.cat([bev_pos_embed, rv_pos_embed], dim=0)
        query_embed = query_embed.transpose(0, 1)  # [num_query, dim] -> [num_query, bs, dim]
        mask =  memory.new_zeros(bs, memory.shape[0]) # [bs, n, h, w] -> [bs, n*h*w]

        target = torch.zeros_like(query_embed)
        # out_dec: [num_layers, num_query, bs, dim]
        out_dec = self.decoder(
            query=target,
            key=memory,
            value=memory,
            key_pos=pos_embed,
            query_pos=query_embed,
            key_padding_mask=mask,
            attn_masks=[attn_masks, None],
            reg_branch=reg_branch,
            )
        out_dec = out_dec.transpose(1, 2)
        return  out_dec, memory


@MODELS.register_module()
class CmtLidarTransformer(BaseModule):
    """Implements the DETR transformer.
    Following the official DETR implementation, this module copy-paste
    from torch.nn.Transformer with modifications:
        * positional encodings are passed in MultiheadAttention
        * extra LN at the end of encoder is removed
        * decoder returns a stack of activations from all decoding layers
    See `paper: End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.
    Args:
        encoder (`mmcv.ConfigDict` | Dict): Config of
            TransformerEncoder. Defaults to None.
        decoder ((`mmcv.ConfigDict` | Dict)): Config of
            TransformerDecoder. Defaults to None
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Defaults to None.
    """

    def __init__(self, encoder=None, decoder=None, init_cfg=None, cross=False):
        super(CmtLidarTransformer, self).__init__(init_cfg=init_cfg)
        if encoder is not None:
            self.encoder = MODELS.build(encoder)
        else:
            self.encoder = None
        self.decoder = MODELS.build(decoder)
        self.embed_dims = self.decoder.embed_dims
        self.cross = cross

    def init_weights(self):
        # follow the official DETR to init parameters
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                xavier_init(m, distribution='uniform')
        self._is_init = True

    def forward(self, x, mask, query_embed, pos_embed, attn_masks=None, reg_branch=None):
        """Forward function for `Transformer`.
        Args:
            x (Tensor): Input query with shape [bs, c, h, w] where
                c = embed_dims.
            mask (Tensor): The key_padding_mask used for encoder and decoder,
                with shape [bs, h, w].
            query_embed (Tensor): The query embedding for decoder, with shape
                [num_query, c].
            pos_embed (Tensor): The positional encoding for encoder and
                decoder, with the same shape as `x`.
        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.
                - out_dec: Output from decoder. If return_intermediate_dec \
                      is True output has shape [num_dec_layers, bs,
                      num_query, embed_dims], else has shape [1, bs, \
                      num_query, embed_dims].
                - memory: Output results from encoder, with shape \
                      [bs, embed_dims, h, w].
        """
        bs, c, h, w = x.shape
        memory = rearrange(x, "bs c h w -> (h w) bs c") # [bs, n, c, h, w] -> [n*h*w, bs, c]
        pos_embed = pos_embed.unsqueeze(1).repeat(1, bs, 1) # [bs, n, c, h, w] -> [n*h*w, bs, c]
        query_embed = query_embed.transpose(0, 1)  # [num_query, dim] -> [num_query, bs, dim]
        mask = mask.view(bs, -1)  # [bs, n, h, w] -> [bs, n*h*w]
        target = torch.zeros_like(query_embed)
        # out_dec: [num_layers, num_query, bs, dim]
        out_dec = self.decoder(
            query=target,
            key=memory,
            value=memory,
            key_pos=pos_embed,
            query_pos=query_embed,
            key_padding_mask=mask,
            attn_masks=[attn_masks, None],
            reg_branch=reg_branch,
            )
        out_dec = out_dec.transpose(1, 2)
        return  out_dec, memory


@MODELS.register_module()
class CmtImageTransformer(BaseModule):
    """Implements the DETR transformer.
    Following the official DETR implementation, this module copy-paste
    from torch.nn.Transformer with modifications:
        * positional encodings are passed in MultiheadAttention
        * extra LN at the end of encoder is removed
        * decoder returns a stack of activations from all decoding layers
    See `paper: End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.
    Args:
        encoder (`mmcv.ConfigDict` | Dict): Config of
            TransformerEncoder. Defaults to None.
        decoder ((`mmcv.ConfigDict` | Dict)): Config of
            TransformerDecoder. Defaults to None
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Defaults to None.
    """

    def __init__(self, encoder=None, decoder=None, init_cfg=None, cross=False):
        super(CmtImageTransformer, self).__init__(init_cfg=init_cfg)
        if encoder is not None:
            self.encoder = MODELS.build(encoder)
        else:
            self.encoder = None
        self.decoder = MODELS.build(decoder)
        self.embed_dims = self.decoder.embed_dims
        self.cross = cross

    def init_weights(self):
        # follow the official DETR to init parameters
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                xavier_init(m, distribution='uniform')
        self._is_init = True

    def forward(self, x_img, query_embed, rv_pos_embed, attn_masks=None, reg_branch=None, bs=2):
        """Forward function for `Transformer`.
        Args:
            x (Tensor): Input query with shape [bs, c, h, w] where
                c = embed_dims.
            mask (Tensor): The key_padding_mask used for encoder and decoder,
                with shape [bs, h, w].
            query_embed (Tensor): The query embedding for decoder, with shape
                [num_query, c].
            pos_embed (Tensor): The positional encoding for encoder and
                decoder, with the same shape as `x`.
        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.
                - out_dec: Output from decoder. If return_intermediate_dec \
                      is True output has shape [num_dec_layers, bs,
                      num_query, embed_dims], else has shape [1, bs, \
                      num_query, embed_dims].
                - memory: Output results from encoder, with shape \
                      [bs, embed_dims, h, w].
        """
        memory = rearrange(x_img, "(bs v) c h w -> (v h w) bs c", bs=bs)
        pos_embed = rearrange(rv_pos_embed, "(bs v) h w c -> (v h w) bs c", bs=bs)
        
        query_embed = query_embed.transpose(0, 1)  # [num_query, dim] -> [num_query, bs, dim]
        mask =  memory.new_zeros(bs, memory.shape[0]) # [bs, n, h, w] -> [bs, n*h*w]

        target = torch.zeros_like(query_embed)
        # out_dec: [num_layers, num_query, bs, dim]
        out_dec = self.decoder(
            query=target,
            key=memory,
            value=memory,
            key_pos=pos_embed,
            query_pos=query_embed,
            key_padding_mask=mask,
            attn_masks=[attn_masks, None],
            reg_branch=reg_branch,
            )
        out_dec = out_dec.transpose(1, 2)
        return  out_dec, memory
