

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
import ml_collections
import numpy as np

from .swin_transformer import SwinTransformer
from .segformer import MiT, LayerNorm
from .resnet import Resnet
from .UNet import UNet
from .CSNet import CSNet
from .munet import MUNet
from .transunet import VisionTransformer
from .swinunet import SwinUnet
from .mmseg_models.mit import MixVisionTransformer
from .mmseg_models.swin import MMSEG_SwinTransformer


def norm_targets(targets, patch_size):
    assert patch_size % 2 == 1
    
    targets_ = targets
    targets_count = torch.ones_like(targets)

    targets_square = targets ** 2.
    
    targets_mean = F.avg_pool2d(targets, kernel_size=patch_size, stride=1, padding=patch_size // 2, count_include_pad=False)
    targets_square_mean = F.avg_pool2d(targets_square, kernel_size=patch_size, stride=1, padding=patch_size // 2, count_include_pad=False)
    targets_count = F.avg_pool2d(targets_count, kernel_size=patch_size, stride=1, padding=patch_size // 2, count_include_pad=True) * (patch_size ** 2)
    
    targets_var = (targets_square_mean - targets_mean ** 2.) * (targets_count / (targets_count - 1))
    targets_var = torch.clamp(targets_var, min=0.)
    
    targets_ = (targets_ - targets_mean) / (targets_var + 1.e-6) ** 0.5
    
    return targets_


# -----------------------------------------------------------------------------
# ----------------------------swin transformer--------------------------------------
# -----------------------------------------------------------------------------
class SwinTransformerForContextEncoder(SwinTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert self.num_classes == 0

    def forward(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = x.transpose(1, 2)
        B, C, L = x.shape
        H = W = int(L ** 0.5)
        x = x.reshape(B, C, H, W)

        return x


    @torch.jit.ignore
    def no_weight_decay(self):
        return super().no_weight_decay() | {'mask_token'}

class SwinTransformerForPixelEncoder(SwinTransformer):
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


# -----------------------------------------------------------------------------
# ----------------------------segformer--------------------------------------
# -----------------------------------------------------------------------------

class SegformerForContextEncoder(MiT):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def forward(
        self,
        x,
    ):
        if self.use_checkpoint:
            ret = checkpoint.checkpoint(self._forward, x)
        else:
            ret = self._forward(x)
        return ret

    def _forward(
        self,
        x, 
    ):
        h, w = x.shape[-2:]

        for (get_overlap_patches, overlap_embed, layers) in self.stages:
            x = get_overlap_patches(x)
            num_patches = x.shape[-1]
            ratio = int(sqrt((h * w) / num_patches))
            x = rearrange(x, 'b c (h w) -> b c h w', h = h // ratio)
            x = overlap_embed(x)

            for (attn, ff) in layers:
                x = attn(x) + x
                x = ff(x) + x

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




# -----------------------------------------------------------------------------
# ----------------------------deeplab resnet--------------------------------------
# -----------------------------------------------------------------------------

class DeeplabResnetForContextEncoder(Resnet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def forward(self, x):
        x = self.inc(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x



class DeeplabResnetForPixelEncoder(Resnet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.norm = nn.BatchNorm2d(self.num_features)

        self.mask_token = nn.Parameter(torch.zeros(1, self.embed_dim, 1, 1))
        trunc_normal_(self.mask_token, mean=0., std=.02)

    def forward(
        self,
        x,
        mask
    ):
        x = self.inc(x)

        b, c, h_, w_ = x.shape
        mask_tokens = self.mask_token.expand(b, -1, h_, w_)
        multi = mask.unsqueeze(1).type_as(x)
        x = x * (1. - multi) + mask_tokens * multi

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        
        return self.norm(x)



# -----------------------------------------------------------------------------
# ----------------------------unet resnet--------------------------------------
# -----------------------------------------------------------------------------

class UNetResnetForContextEncoder(UNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        del self.up1
        del self.up2
        del self.up3
        del self.up4
        del self.outc

        del self.down4.maxpool_conv[-1].double_conv[-3]



    def forward(self, x):
        x = self.inc(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        return x



class UNetResnetForPixelEncoder(UNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        del self.up1
        del self.up2
        del self.up3
        del self.up4
        del self.outc

        # self.norm = nn.BatchNorm2d(self.num_features)

        self.mask_token = nn.Parameter(torch.zeros(1, self.embed_dim, 1, 1))
        trunc_normal_(self.mask_token, mean=0., std=.02)

    def forward(
        self,
        x,
        mask
    ):
        x = self.inc(x)

        b, c, h_, w_ = x.shape
        mask_tokens = self.mask_token.expand(b, -1, h_, w_)
        multi = mask.unsqueeze(1).type_as(x)
        x = x * (1. - multi) + mask_tokens * multi

        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        
        return x



# -----------------------------------------------------------------------------
# ----------------------------csnet--------------------------------------
# -----------------------------------------------------------------------------

class CSnetForContextEncoder(CSNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        del self.up1
        del self.up2
        del self.up3
        del self.up4
        del self.outc


    def forward(self, x):
        x = self.inc(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        return x

class CSnetForPixelEncoder(CSNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        del self.up1
        del self.up2
        del self.up3
        del self.up4
        del self.outc

        self.norm = nn.BatchNorm2d(self.num_features)

        self.mask_token = nn.Parameter(torch.zeros(1, self.embed_dim, 1, 1))
        trunc_normal_(self.mask_token, mean=0., std=.02)

    def forward(
        self,
        x,
        mask
    ):
        x = self.inc(x)

        b, c, h_, w_ = x.shape
        mask_tokens = self.mask_token.expand(b, -1, h_, w_)
        multi = mask.unsqueeze(1).type_as(x)
        x = x * (1. - multi) + mask_tokens * multi

        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        
        return self.norm(x)
    

# -----------------------------------------------------------------------------
# ----------------------------mnet--------------------------------------
# -----------------------------------------------------------------------------

class MUnetForContextEncoder(MUNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        del self.up1
        del self.up2
        del self.up3
        del self.up4
        del self.outc
        del self.respath1
        del self.respath2
        del self.down4.maxpool_conv[-1].double_conv[-3]


    def forward(self, x):
        x = self.inc(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        return x

class MUnetForPixelEncoder(MUNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        del self.up1
        del self.up2
        del self.up3
        del self.up4
        del self.outc
        del self.respath1
        del self.respath2

        # self.norm = nn.BatchNorm2d(self.num_features)

        self.mask_token = nn.Parameter(torch.zeros(1, self.embed_dim, 1, 1))
        trunc_normal_(self.mask_token, mean=0., std=.02)

    def forward(
        self,
        x,
        mask
    ):
        x = self.inc(x)

        b, c, h_, w_ = x.shape
        mask_tokens = self.mask_token.expand(b, -1, h_, w_)
        multi = mask.unsqueeze(1).type_as(x)
        x = x * (1. - multi) + mask_tokens * multi

        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        
        return x


# -----------------------------------------------------------------------------
# ----------------------------transunet--------------------------------------
# -----------------------------------------------------------------------------

def get_b16_config():
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1

    config.classifier = 'seg'
    config.representation_size = None
    config.resnet_pretrained_path = None
    config.pretrained_path = '../model/vit_checkpoint/imagenet21k/ViT-B_16.npz'
    config.patch_size = 16

    config.decoder_channels = (256, 128, 64, 16)
    config.n_classes = 2
    config.activation = 'softmax'
    return config

def get_r50_b16_config():
    """Returns the Resnet50 + ViT-B/16 configuration."""
    config = get_b16_config()
    config.patches.grid = (16, 16)
    config.resnet = ml_collections.ConfigDict()
    config.resnet.num_layers = (3, 4, 9)
    config.resnet.width_factor = 1

    config.classifier = 'seg'
    config.pretrained_path = '../model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz'
    config.decoder_channels = (256, 128, 64, 16)
    config.skip_channels = [512, 256, 64, 16]
    config.n_classes = 2
    config.n_skip = 3
    config.activation = 'softmax'

    return config

CONFIGS = {
    'ViT-B_16': get_b16_config(),
    'R50-ViT-B_16': get_r50_b16_config(),
}


class TransunetForContextEncoder(VisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        del self.decoder
        del self.segmentation_head
        self.transformer.encoder.encoder_norm = nn.Identity()


    def forward(self, x):
        x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)

        B, n_patch, hidden = x.size()
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = x.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        return x

class TransunetForPixelEncoder(VisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        del self.decoder
        del self.segmentation_head

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        trunc_normal_(self.mask_token, mean=0., std=.02)

    def forward(
        self,
        x,
        mask
    ):
        x, _ = self.transformer.embeddings(x)

        b, n, c = x.shape
        mask_tokens = self.mask_token.expand(b, n, -1)
        multi = mask.flatten(1).unsqueeze(-1).type_as(x)
        x = x * (1. - multi) + mask_tokens * multi

        x, _ = self.transformer.encoder(x)  # (B, n_patch, hidden)

        B, n_patch, hidden = x.size()
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = x.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        return x

# -----------------------------------------------------------------------------
# ----------------------------swinunet--------------------------------------
# -----------------------------------------------------------------------------

class SwinunetForContextEncoder(SwinUnet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        del self.up1
        del self.up2
        del self.up3
        del self.invemb
        del self.outc

    def forward(self, x):
        x_list = self.encoder(x)  # (B, n_patch, hidden)
        x = x_list[-1]
        x = x.contiguous()
        return x
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


# -----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------MMSegmentation for CityScape dataset-------------------------------------
# -----------------------------------------------------------------------------------------------------------------------

# ------------------segformer encoder -> mix vision transformer-------------------
class MMSegMiTForContextEncoder(MixVisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    
class MMSegMiTForPixelEncoder(MixVisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        trunc_normal_(self.mask_token, mean=0., std=.02)

    def forward(self, x, mask):

        for i, layer in enumerate(self.layers):
            x, hw_shape = layer[0](x)
            if i == 0:
                b, n, c = x.shape
                mask_tokens = self.mask_token.expand(b, n, -1)
                multi = mask.flatten(1).unsqueeze(-1).type_as(x)
                x = x * (1. - multi) + mask_tokens * multi

            for block in layer[1]:
                x = block(x, hw_shape)
            
            x = layer[2](x)
            x = nlc_to_nchw(x, hw_shape)

        return x

# ------------------upernet encoder -> swin transformer-------------------
class MMSegSwinForContextEncoder(MMSEG_SwinTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    
class MMSegSwinForPixelEncoder(MMSEG_SwinTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        trunc_normal_(self.mask_token, mean=0., std=.02)

    def forward(self, x, mask):

        x, hw_shape = self.patch_embed(x)

        b, n, c = x.shape
        mask_tokens = self.mask_token.expand(b, n, -1)
        multi = mask.flatten(1).unsqueeze(-1).type_as(x)
        x = x * (1. - multi) + mask_tokens * multi

        if self.use_abs_pos_embed:
            x = x + self.absolute_pos_embed
        x = self.drop_after_pos(x)

        for i, stage in enumerate(self.stages):
            x, hw_shape, out, out_hw_shape = stage(x, hw_shape)


        x = self.norm3(x).view(-1, *out_hw_shape,
                               self.num_features).permute(0, 3, 1,
                                                             2).contiguous()

        return x
# --------------------------------useful tools--------------------------------
def nlc_to_nchw(x, hw_shape):
    """Convert [N, L, C] shape tensor to [N, C, H, W] shape tensor.

    Args:
        x (Tensor): The input tensor of shape [N, L, C] before conversion.
        hw_shape (Sequence[int]): The height and width of output feature map.

    Returns:
        Tensor: The output tensor of shape [N, C, H, W] after conversion.
    """
    H, W = hw_shape
    assert len(x.shape) == 3
    B, L, C = x.shape
    assert L == H * W, 'The seq_len doesn\'t match H, W'
    return x.transpose(1, 2).reshape(B, C, H, W)


# -----------------------------------------------------------------------------
# ----------------------------mask pixel and context representation--------------------------------------
# -----------------------------------------------------------------------------

class PixelDecoder(nn.Module):
    def __init__(self, num_features, encoder_stride):
        super().__init__()
        self.pixel_decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=num_features,   # the dimension of encoder output
                out_channels=encoder_stride ** 2 * 3, kernel_size=1),  
            nn.PixelShuffle(encoder_stride),
        )
        self.pixel_q = nn.Conv2d(
                in_channels=num_features,   # the dimension of encoder output
                out_channels=num_features, kernel_size=1)
    def forward(self, x):
        x_rec_intermediate = self.decoder[0](x)
        x_rec = self.decoder[-1](x_rec_intermediate)

        pixel_q = self.pixel_q(x_rec_intermediate)
        return x_rec, pixel_q

class SementicDecoder(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.sementic_kv = nn.Conv2d(
                in_channels=num_features,   # the dimension of encoder output
                out_channels=num_features, kernel_size=1)
        
        self.cross_attention = CrossAttentionAlign(dim=num_features, num_heads=8)
    def forward(self, z, q):
        z_kv = self.sementic_kv(z)

        z_context_rec = self.cross_attention(q, z_kv)
        return z_context_rec


class JointDecoder(nn.Module):
    def __init__(self, num_features, encoder_stride):
        super().__init__()

        self.pixel_decoder = PixelDecoder(num_features=num_features, encoder_stride=encoder_stride)

        self.sementic_decoder = SementicDecoder(num_features=num_features)

    def forward(self, x):
        # pixel encoder
        y_p, x_rec_q = self.pixel_decoder(x)
        # sementic encoder
        y_s = self.sementic_decoder(x, x_rec_q)  # [b, c, h=6, w=6]

        return y_p, y_s


class MaskedPixelSemanticReconstruction(nn.Module):
    def __init__(self, encoder, context_encoder, encoder_stride, in_chans, patch_size):
        super().__init__()
        self.encoder = encoder
        self.encoder_stride = encoder_stride

        self.sementic_encoder = context_encoder
        for param in self.sementic_encoder.parameters():
            param.requires_grad = False

        self.joint_decoder = JointDecoder(num_features=self.encoder.num_features, encoder_stride=self.encoder_stride)

        self.in_chans = in_chans
        self.patch_size = patch_size
        self.scale = self.encoder_stride // self.patch_size

    def forward(self, x, mask):
        z = self.encoder(x, mask)
        z_context = self.sementic_encoder(x)

        x_rec, z_context_rec = self.joint_decoder(z)
        
        # create mask 
        mask_pixel = mask.repeat_interleave(self.patch_size, 1).repeat_interleave(self.patch_size, 2).unsqueeze(1).contiguous()
        mask_context = mask[:, ::self.scale, ::self.scale].unsqueeze(1).contiguous()  # [b, 1, h=6, w=6]

        # loss = L2(x, x_rec) * mask + L2(z_context, z_context_rec) * mask_context
        loss_recon = F.mse_loss(x, x_rec, reduction='none')
        loss_pixel = (loss_recon * mask_pixel).sum() / (mask_pixel.sum() + 1e-5) / self.in_chans

        loss_recon_context = F.mse_loss(z_context, z_context_rec, reduction='none')
        loss_context = (loss_recon_context * mask_context).sum() / (mask_context.sum() + 1e-5) / self.encoder.num_features
        return loss_pixel, loss_context
    

    def update_target(self, m):
        with torch.no_grad():
            encoder_state_dict = self.encoder.state_dict()
            
            for name, param_k in self.context_encoder.named_parameters():
                if name in encoder_state_dict:
                    param_q = encoder_state_dict[name]
                    param_k.data.mul_(m).add_((1. - m) * param_q)

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


class CrossAttentionAlign(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        
    def forward(self, z, z_context):
        """
        z: [B, C, H, W] (encoder特征) q
        z_context: [B, C, H, W] (context特征) kv
        """
        B, C, H, W = z.shape
        z = z.flatten(2).transpose(1, 2)          # [B, HW, C]
        z_context = z_context.flatten(2).transpose(1, 2)  # [B, HW, C]

        # Cross-Attention: query = z, key/value = z_context
        z_aligned, _ = self.cross_attn(query=z, key=z_context, value=z_context)

        z_aligned = z_aligned.transpose(1, 2).view(B, C, H, W)

        return z_aligned

def build_mpsr(config):
    model_type = config.MODEL.TYPE
    if model_type == 'swin_mpsr':
        pixel_encoder = SwinTransformerForPixelEncoder(
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
        context_encoder = SwinTransformerForContextEncoder(
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
        in_chans = config.MODEL.SWIN.IN_CHANS
        patch_size = config.MODEL.SWIN.PATCH_SIZE
        
    elif model_type == 'swinunet_mpsr':
        pixel_encoder = SwinunetForPixelEncoder(
            img_size=config.DATA.IMG_SIZE,
            window_size=config.MODEL.SWIN.WINDOW_SIZE,
            embed_dim=config.MODEL.SWIN.EMBED_DIM,
            use_shifted_window=True,
            use_relat_position=True,
            )
        context_encoder = SwinunetForContextEncoder(
            img_size=config.DATA.IMG_SIZE,
            window_size=config.MODEL.SWIN.WINDOW_SIZE,
            embed_dim=config.MODEL.SWIN.EMBED_DIM,
            use_shifted_window=True,
            use_relat_position=True,)
        encoder_stride = 32
        in_chans = 3
        patch_size = 4
    
    elif model_type == 'segformer_mpsr':
        pixel_encoder = SegformerForPixelEncoder(
            channels=config.MODEL.SEGFORMER.IN_CHANS,
            dims=config.MODEL.SEGFORMER.DIMS, #(32, 64, 128, 256)
            heads=config.MODEL.SEGFORMER.HEADS, #(1, 1, 1, 1),
            ff_expansion=config.MODEL.SEGFORMER.FF_EXPANSION, #(4, 4, 4, 4),
            reduction_ratio=config.MODEL.SEGFORMER.REDCTION_RATIO, #(1, 1, 1, 1),
            num_layers=config.MODEL.SEGFORMER.NUM_LAYERS, #(4, 4, 4, 4),
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
            )
        context_encoder = SegformerForContextEncoder(
            channels=config.MODEL.SEGFORMER.IN_CHANS,
            dims=config.MODEL.SEGFORMER.DIMS, #(32, 64, 128, 256)
            heads=config.MODEL.SEGFORMER.HEADS, #(1, 1, 1, 1),
            ff_expansion=config.MODEL.SEGFORMER.FF_EXPANSION, #(4, 4, 4, 4),
            reduction_ratio=config.MODEL.SEGFORMER.REDCTION_RATIO, #(1, 1, 1, 1),
            num_layers=config.MODEL.SEGFORMER.NUM_LAYERS, #(4, 4, 4, 4),
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
            )
        encoder_stride = 32
        in_chans = config.MODEL.SEGFORMER.IN_CHANS
        patch_size = config.MODEL.SEGFORMER.PATCH_SIZE


    elif model_type == 'deeplabresnet_mpsr':
        pixel_encoder = DeeplabResnetForPixelEncoder(
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            )
        context_encoder = DeeplabResnetForContextEncoder(
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            )
        encoder_stride = 16
        in_chans = 3
        patch_size = 1 # 1

    elif model_type == 'unetresnet_mpsr':
        pixel_encoder = UNetResnetForPixelEncoder(
            n_classes=4,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            )
        context_encoder = UNetResnetForContextEncoder(
            n_classes=4,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            )
        encoder_stride = 16
        in_chans = 3
        patch_size = 1 # 1

    elif model_type == 'csnet_mpsr':
        pixel_encoder = CSnetForPixelEncoder(
            n_classes=4,
            num_heads=config.MODEL.CSNET.NUM_HEADS,
            choice=config.MODEL.CSNET.CHOICE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            patch_size=config.MODEL.CSNET.PATCH_SIZE,
            )
        context_encoder = CSnetForContextEncoder(
            n_classes=4,
            num_heads=config.MODEL.CSNET.NUM_HEADS,
            choice=config.MODEL.CSNET.CHOICE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            patch_size=config.MODEL.CSNET.PATCH_SIZE,
            )
        encoder_stride = 16
        in_chans = 3
        patch_size = 1 # 1

    elif model_type == 'munet_mpsr':
        pixel_encoder = MUnetForPixelEncoder(
            n_classes=4,
            )
        context_encoder = MUnetForContextEncoder(
            n_classes=4,
            )
        encoder_stride = 16
        in_chans = 3
        patch_size = 1 # 1

    elif model_type == 'transunet_mpsr':
        transunet_name = 'R50-ViT-B_16'
        vit_patches_size = 16
        config_vit = CONFIGS[transunet_name]
        img_size = config.DATA.IMG_SIZE
        if transunet_name.find('R50') != -1:
            config_vit.patches.grid = (int(img_size / vit_patches_size), int(img_size / vit_patches_size))
        
        pixel_encoder = TransunetForPixelEncoder(
            config=config_vit,
            img_size=config.DATA.IMG_SIZE,
            num_classes=4,
            )
        context_encoder = TransunetForContextEncoder(
            config=config_vit,
            img_size=config.DATA.IMG_SIZE,
            num_classes=4,
            )
        encoder_stride = vit_patches_size   # =patch_size*downscale*downscale*downscale = 16 * 1 * 1 * 1
        in_chans = 3
        patch_size = vit_patches_size # 16

    
    # ---------------------------MMSEG------------------------------------
    elif model_type == 'mmsegmit_mpsr':
        pixel_encoder = MMSegMiTForPixelEncoder(
            embed_dims=config.MODEL.MIT.EMBED_DIMS, # 64
            num_heads=config.MODEL.MIT.NUM_HEADS, # [1, 2, 5, 8]
            num_layers=config.MODEL.MIT.NUM_LAYERS, # [3, 4, 6, 3],

            in_channels=3,
            num_stages=4,
            strides=[4, 2, 2, 2],
            patch_sizes=[7, 3, 3, 3],
            sr_ratios=[8, 4, 2, 1],
            out_indices=(0, 1, 2, 3),
            mlp_ratio=4,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
            )
        context_encoder = MMSegMiTForContextEncoder(
            embed_dims=config.MODEL.MIT.EMBED_DIMS, # 64
            num_heads=config.MODEL.MIT.NUM_HEADS, # [1, 2, 5, 8]
            num_layers=config.MODEL.MIT.NUM_LAYERS, # [3, 4, 6, 3],

            in_channels=3,
            num_stages=4,
            strides=[4, 2, 2, 2],
            patch_sizes=[7, 3, 3, 3],
            sr_ratios=[8, 4, 2, 1],
            out_indices=(0, 1, 2, 3),
            mlp_ratio=4,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
            )
        encoder_stride = 32
        in_chans = 3
        patch_size = 4

    elif model_type == 'mmsegswin_mpsr':
        backbone_norm_cfg = dict(type='LN', requires_grad=True)

        pixel_encoder = MMSegSwinForPixelEncoder(
            pretrain_img_size=config.DATA.IMG_SIZE,  # 224
            window_size=config.MODEL.SWIN.WINDOW_SIZE, # 7
            embed_dims=config.MODEL.SWIN.EMBED_DIM, # 64
            depths=config.MODEL.SWIN.DEPTHS, # [2, 2, 6, 2]
            num_heads=config.MODEL.SWIN.NUM_HEADS, # [3, 6, 12, 24]

            patch_size=4,
            mlp_ratio=4,
            strides=(4, 2, 2, 2),
            out_indices=(0, 1, 2, 3),
            qkv_bias=True,
            qk_scale=None,
            patch_norm=True,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.3,
            use_abs_pos_embed=False,
            act_cfg=dict(type='GELU'),
            norm_cfg=backbone_norm_cfg
            )
        context_encoder = MMSegSwinForContextEncoder(
            pretrain_img_size=config.DATA.IMG_SIZE,  # 224
            window_size=config.MODEL.SWIN.WINDOW_SIZE, # 7
            embed_dims=config.MODEL.SWIN.EMBED_DIM, #64
            depths=config.MODEL.SWIN.DEPTHS, # [2, 2, 6, 2]
            num_heads=config.MODEL.SWIN.NUM_HEADS, # [3, 6, 12, 24]

            patch_size=4,
            mlp_ratio=4,
            strides=(4, 2, 2, 2),
            out_indices=(0, 1, 2, 3),
            qkv_bias=True,
            qk_scale=None,
            patch_norm=True,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.3,
            use_abs_pos_embed=False,
            act_cfg=dict(type='GELU'),
            norm_cfg=backbone_norm_cfg
            )
        encoder_stride = 32
        in_chans = 3
        patch_size = 4

    else:
        raise NotImplementedError(f"Unknown pre-train model: {model_type}")
    
    model = MaskedPixelSemanticReconstruction(
                    encoder=pixel_encoder, 
                    context_encoder=context_encoder, 
                    encoder_stride=encoder_stride, 
                    in_chans=in_chans, 
                    patch_size=patch_size)

    return model


if __name__ == '__main__':

    pixel_encoder = SwinunetForPixelEncoder(
            img_size=256,
            window_size=8,
            embed_dim=64,
            use_shifted_window=True,
            use_relat_position=True,
            )
    context_encoder = SwinunetForContextEncoder(
            img_size=256,
            window_size=8,
            embed_dim=64,
            use_shifted_window=True,
            use_relat_position=True,)
    encoder_stride = 32
    in_chans = 3
    patch_size = 4
    model = MaskedPixelSemanticReconstruction(
                    encoder=pixel_encoder, 
                    context_encoder=context_encoder, 
                    encoder_stride=encoder_stride, 
                    in_chans=in_chans, 
                    patch_size=patch_size)
    
    print(model)


