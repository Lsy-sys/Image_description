"""
Vision Transformer (ViT) 编码器
用于Model D
"""

import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig


class ViTEncoder(nn.Module):
    """Vision Transformer编码器"""
    
    def __init__(self, model_name='google/vit-base-patch16-224', 
                 feature_dim=768, patch_size=16):
        """
        Args:
            model_name: 预训练模型名称
            feature_dim: 特征维度
            patch_size: patch大小
        """
        super().__init__()
        
        # 加载预训练的ViT模型
        self.vit = ViTModel.from_pretrained(model_name)
        self.feature_dim = feature_dim
        
        # 如果特征维度不匹配，添加投影层
        if self.vit.config.hidden_size != feature_dim:
            self.projection = nn.Linear(self.vit.config.hidden_size, feature_dim)
        else:
            self.projection = nn.Identity()
    
    def forward(self, images: torch.Tensor):
        """
        前向传播
        Args:
            images: 输入图像 (batch_size, 3, H, W)
        Returns:
            图像特征 (batch_size, num_patches, feature_dim)
        """
        # ViT前向传播
        outputs = self.vit(pixel_values=images)
        
        # 获取序列输出（包含[CLS] token和patch tokens）
        sequence_output = outputs.last_hidden_state  # (batch_size, num_patches+1, hidden_size)
        
        # 投影到目标维度
        features = self.projection(sequence_output)  # (batch_size, num_patches+1, feature_dim)
        
        return features



