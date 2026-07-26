"""
ARNIQA Module

This module provides:
1. Image degradation generation capabilities for training and evaluation
2. Quality-aware conditioning for the restoration U-Net (Stage 1 and 2)

Components:
- ImageDistorter: Degradation generation (25 types, 5 severity levels)
- ArniqaEncoder: ResNet-50 backbone for quality feature extraction (frozen)
- ArniqaGlobalAdapter: Projects global features to cross-attention format
- ArniqaSpatialAdapter: Projects Layer 3 spatial features to tokens (Stage 2)
- ArniqaConditioner: Complete conditioning module (encoder + adapters)
- get_2d_sincos_pos_embed: Utility for 2D sinusoidal positional embeddings
"""

from .degradation import ImageDistorter
from .model import (
    ArniqaEncoder, 
    ArniqaGlobalAdapter, 
    ArniqaSpatialAdapter,
    ArniqaConditioner,
    get_2d_sincos_pos_embed,
)


__all__ = [
    'ImageDistorter',
    'ArniqaEncoder',
    'ArniqaGlobalAdapter',
    'ArniqaSpatialAdapter',
    'ArniqaConditioner',
    'get_2d_sincos_pos_embed',
]
