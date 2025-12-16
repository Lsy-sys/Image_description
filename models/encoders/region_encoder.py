"""
Faster R-CNN区域编码器
用于提取图像中的区域特征（用于Model C）
"""

import torch
import torch.nn as nn
from typing import Optional
import torchvision.models as models


class FasterRCNNEncoder(nn.Module):
    """
    Faster R-CNN 区域编码器 / 兼容图像输入的区域编码器

    当前项目中，Region-Transformer 直接从原始图像开始训练，
    因此这里做了一个折中：

    - 如果输入是 4D 图像张量 (B, 3, H, W)，先通过 ResNet50 backbone
      做全局池化，得到 (B, 1, 2048) 作为「单区域特征」；
    - 如果输入已经是 3D 区域特征 (B, num_regions, feature_dim)，
      则直接视为预提取的区域特征。
    """
    
    def __init__(self, feature_dim: int = 512, max_regions: int = 36):
        """
        Args:
            feature_dim: 区域特征维度（默认 2048，对应 ResNet50 输出）
            max_regions: 最大区域数量（暂未强制使用，保留扩展）
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.max_regions = max_regions

        # 使用 ResNet50 作为图像 backbone，将 (B, 3, H, W) -> (B, 2048, 1, 1)
        resnet = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # 去掉 FC 层

        # ResNet 输出维度固定为 2048，这里统一投影到 feature_dim（需与 Transformer d_model 对齐）
        backbone_dim = 2048
        self.projection = nn.Linear(backbone_dim, feature_dim)
        self.norm = nn.LayerNorm(feature_dim)
    
    def forward(self, regions: torch.Tensor):
        """
        前向传播
        Args:
            regions:
                - 图像张量: (batch_size, 3, H, W)
                - 区域特征: (batch_size, num_regions, feature_dim)
        Returns:
            区域特征: (batch_size, num_regions, feature_dim)
        """
        if not isinstance(regions, torch.Tensor):
            raise TypeError("FasterRCNNEncoder 期望输入为 torch.Tensor")

        # 情况一：来自 DataLoader 的原始图像 (B, 3, H, W)
        if regions.dim() == 4:
            # 通过 ResNet backbone 提取全局特征 (B, 2048, 1, 1)
            feats = self.backbone(regions)
            feats = feats.view(feats.size(0), 1, -1)  # (B, 1, 2048)
            features = feats

        # 情况二：已经是区域特征 (B, num_regions, D)，无论 D 是 2048 还是 feature_dim，都统一投影
        elif regions.dim() == 3:
            # 将最后一维投影到 feature_dim
            features = regions

        else:
            raise ValueError(
                f"FasterRCNNEncoder 收到的张量形状不支持: {regions.shape}，"
                f"期望 (B, 3, H, W) 或 (B, num_regions, {self.feature_dim})"
            )
        
        # 线性投影 + LayerNorm
        features = self.projection(features)
        features = self.norm(features)
        
        return features





