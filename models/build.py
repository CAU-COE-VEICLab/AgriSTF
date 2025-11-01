# --------------------------------------------------------
# MPSR: Masked Pixel-Semantic Reconstruction
# Copyright (c) 2025 CAU
# Licensed under The MIT License [see LICENSE for details]
# Written by Guorun Li and Yucong Wang
# --------------------------------------------------------

from .swin_transformer import build_swin
from .vision_transformer import build_vit
from .simmim import build_simmim
from .mpsr import build_mpsr


def build_model(config, is_pretrain=True):
    if is_pretrain:
        model_type = config.MODEL.TYPE
        if model_type.endswith('mpsr'):
            model = build_mpsr(config)
        else:
            model = build_simmim(config)
    else:
        model_type = config.MODEL.TYPE
        if model_type == 'swin':
            model = build_swin(config)
        elif model_type == 'vit':
            model = build_vit(config)
        else:
            raise NotImplementedError(f"Unknown fine-tune model: {model_type}")

    return model

