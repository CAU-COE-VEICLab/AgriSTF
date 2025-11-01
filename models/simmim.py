# --------------------------------------------------------
# SimMIM
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Zhenda Xie
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
import torch.utils.checkpoint as checkpoint
from math import sqrt

from .swin_transformer import SwinTransformer
from .vision_transformer import VisionTransformer
from .segformer import MiT, LayerNorm
from .swinunet import SwinUnet

class SwinTransformerForSimMIM(SwinTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert self.num_classes == 0

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        trunc_normal_(self.mask_token, mean=0., std=.02)

    def forward(self, x, mask):
        x = self.patch_embed(x)

        assert mask is not None
        B, L, _ = x.shape

        mask_tokens = self.mask_token.expand(B, L, -1)
        w = mask.flatten(1).unsqueeze(-1).type_as(mask_tokens)
        x = x * (1. - w) + mask_tokens * w

        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)

        x = x.transpose(1, 2)
        B, C, L = x.shape
        H = W = int(L ** 0.5)
        x = x.reshape(B, C, H, W)
        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return super().no_weight_decay() | {'mask_token'}


class VisionTransformerForSimMIM(VisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert self.num_classes == 0

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self._trunc_normal_(self.mask_token, std=.02)

    def _trunc_normal_(self, tensor, mean=0., std=1.):
        trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)

    def forward(self, x, mask):
        x = self.patch_embed(x)

        assert mask is not None
        B, L, _ = x.shape

        mask_token = self.mask_token.expand(B, L, -1)
        w = mask.flatten(1).unsqueeze(-1).type_as(mask_token)
        x = x * (1 - w) + mask_token * w

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)

        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:
            x = blk(x, rel_pos_bias=rel_pos_bias)
        x = self.norm(x)

        x = x[:, 1:]
        B, L, C = x.shape
        H = W = int(L ** 0.5)
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        return x



class SegformerForPixelEncoder(MiT):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.norm = LayerNorm(dim = self.num_features)

        self.mask_token = nn.Parameter(torch.zeros(1, self.embed_dim, 1, 1))
        trunc_normal_(self.mask_token, mean=0., std=.02)

    def forward(
        self,
        x,
        mask
    ):
        if self.use_checkpoint:
            ret = checkpoint.checkpoint(self._forward, x, mask)
        else:
            ret = self._forward(x, mask)
        return ret

    def _forward(
        self,
        x, 
        mask
    ):
        # mask [batch_Size, maks_number*mask_size/patch_size, maks_number*mask_size/patch_size]
        h, w = x.shape[-2:]
        index_ = 0

        for (get_overlap_patches, overlap_embed, layers) in self.stages:
            x = get_overlap_patches(x)
            num_patches = x.shape[-1]
            ratio = int(sqrt((h * w) / num_patches))
            x = rearrange(x, 'b c (h w) -> b c h w', h = h // ratio)
            x = overlap_embed(x)

            if index_ == 0:
                b, c, h_, w_ = x.shape
                mask_tokens = self.mask_token.expand(b, -1, h_, w_)
                multi = mask.unsqueeze(1).type_as(x)
                x = x * (1. - multi) + mask_tokens * multi


            for (attn, ff) in layers:
                x = attn(x) + x
                x = ff(x) + x

            index_ += 1

        return self.norm(x)
    

class SwinunetForPixelEncoder(SwinUnet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        del self.up1
        del self.up2
        del self.up3
        del self.invemb
        del self.outc

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        trunc_normal_(self.mask_token, mean=0., std=.02)

    def forward(
        self,
        x,
        mask
    ):
        x = self.encoder.patch_embed(x)

        b, n, c = x.shape
        mask_tokens = self.mask_token.expand(b, n, -1)
        multi = mask.flatten(1).unsqueeze(-1).type_as(x)
        x = x * (1. - multi) + mask_tokens * multi

        if not self.encoder.use_relat_position:
            x = x + self.encoder.absolute_pos_embed
        x = self.encoder.pos_drop(x)
        stages_feature = []
        for layer in self.encoder.layers:
            x, x_before_downsampling = layer(x)
            x_reshape = self.encoder.change_feature_shape(x_before_downsampling, layer.input_resolution[0],
                                                  layer.input_resolution[1],
                                                  )
        return x_reshape
    
    

class SimMIM(nn.Module):
    def __init__(self, encoder, encoder_stride):
        super().__init__()
        self.encoder = encoder
        self.encoder_stride = encoder_stride

        self.decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=self.encoder.num_features,
                out_channels=self.encoder_stride ** 2 * 3, kernel_size=1),
            nn.PixelShuffle(self.encoder_stride),
        )

        self.in_chans = self.encoder.in_chans
        self.patch_size = self.encoder.patch_size

    def forward(self, x, mask):
        z = self.encoder(x, mask)
        x_rec = self.decoder(z)

        mask = mask.repeat_interleave(self.patch_size, 1).repeat_interleave(self.patch_size, 2).unsqueeze(1).contiguous()
        loss_recon = F.l1_loss(x, x_rec, reduction='none')
        loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / self.in_chans
        return loss

    @torch.jit.ignore
    def no_weight_decay(self):
        if hasattr(self.encoder, 'no_weight_decay'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay()}
        return {}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        if hasattr(self.encoder, 'no_weight_decay_keywords'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay_keywords()}
        return {}


def build_simmim(config):
    model_type = config.MODEL.TYPE
    if model_type == 'swin':
        encoder = SwinTransformerForSimMIM(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.SWIN.PATCH_SIZE,
            in_chans=config.MODEL.SWIN.IN_CHANS,
            num_classes=0,
            embed_dim=config.MODEL.SWIN.EMBED_DIM,
            depths=config.MODEL.SWIN.DEPTHS,
            num_heads=config.MODEL.SWIN.NUM_HEADS,
            window_size=config.MODEL.SWIN.WINDOW_SIZE,
            mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
            qkv_bias=config.MODEL.SWIN.QKV_BIAS,
            qk_scale=config.MODEL.SWIN.QK_SCALE,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            ape=config.MODEL.SWIN.APE,
            patch_norm=config.MODEL.SWIN.PATCH_NORM,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT)
        encoder_stride = 32
    elif model_type == 'vit':
        encoder = VisionTransformerForSimMIM(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.VIT.PATCH_SIZE,
            in_chans=config.MODEL.VIT.IN_CHANS,
            num_classes=0,
            embed_dim=config.MODEL.VIT.EMBED_DIM,
            depth=config.MODEL.VIT.DEPTH,
            num_heads=config.MODEL.VIT.NUM_HEADS,
            mlp_ratio=config.MODEL.VIT.MLP_RATIO,
            qkv_bias=config.MODEL.VIT.QKV_BIAS,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            init_values=config.MODEL.VIT.INIT_VALUES,
            use_abs_pos_emb=config.MODEL.VIT.USE_APE,
            use_rel_pos_bias=config.MODEL.VIT.USE_RPB,
            use_shared_rel_pos_bias=config.MODEL.VIT.USE_SHARED_RPB,
            use_mean_pooling=config.MODEL.VIT.USE_MEAN_POOLING)
        encoder_stride = 16
    elif model_type == 'segformer':
        encoder = SegformerForPixelEncoder(
            channels=config.MODEL.SEGFORMER.IN_CHANS,
            dims=config.MODEL.SEGFORMER.DIMS, #(32, 64, 128, 256)
            heads=config.MODEL.SEGFORMER.HEADS, #(1, 1, 1, 1),
            ff_expansion=config.MODEL.SEGFORMER.FF_EXPANSION, #(4, 4, 4, 4),
            reduction_ratio=config.MODEL.SEGFORMER.REDCTION_RATIO, #(1, 1, 1, 1),
            num_layers=config.MODEL.SEGFORMER.NUM_LAYERS, #(4, 4, 4, 4),
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
            )
        encoder_stride = 32
    elif model_type == 'swinunet':
        encoder = SwinunetForPixelEncoder(
            img_size=config.DATA.IMG_SIZE,
            window_size=config.MODEL.SWIN.WINDOW_SIZE,
            embed_dim=config.MODEL.SWIN.EMBED_DIM,
            use_shifted_window=True,
            use_relat_position=True,
            )
        encoder_stride = 32
    
    else:
        raise NotImplementedError(f"Unknown pre-train model: {model_type}")

    model = SimMIM(encoder=encoder, encoder_stride=encoder_stride)

    return model
