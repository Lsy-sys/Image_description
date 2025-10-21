"""
模型定义模块
包含CNN+GRU和区域特征+Transformer两种模型
"""

from .cnn_gru.model import CNNGruModel
from .transformer.model import TransformerModel

__all__ = [
    'CNNGruModel',
    'TransformerModel'
]
