"""
视觉编码器库
"""

from .cnn_encoder import ResNetEncoder
from .region_encoder import FasterRCNNEncoder
from .vit_encoder import ViTEncoder
from .graph_encoder import GCNEncoder

__all__ = [
    'ResNetEncoder',
    'FasterRCNNEncoder',
    'ViTEncoder',
    'GCNEncoder'
]

















