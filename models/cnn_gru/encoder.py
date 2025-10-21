"""
CNN编码器
使用预训练的ResNet提取图像特征
"""

import torch
import torch.nn as nn
import torchvision.models as models


class CNNEncoder(nn.Module):
    """CNN编码器"""
    
    def __init__(self, embed_size=512, pretrained=True):
        """
        Args:
            embed_size: 嵌入维度
            pretrained: 是否使用预训练权重
        """
        super(CNNEncoder, self).__init__()
        
        # 使用ResNet-50作为backbone
        resnet = models.resnet50(pretrained=pretrained)
        
        # 移除最后的全连接层和平均池化层
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        
        # 添加自适应平均池化
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 添加线性层映射到embed_size
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        
    def forward(self, images):
        """
        前向传播
        Args:
            images: 输入图像 (batch_size, 3, H, W)
        Returns:
            图像特征 (batch_size, embed_size)
        """
        with torch.no_grad():
            features = self.resnet(images)  # (batch_size, 2048, H, W)
        
        # 全局平均池化
        features = self.adaptive_pool(features)  # (batch_size, 2048, 1, 1)
        features = features.reshape(features.size(0), -1)  # (batch_size, 2048)
        
        # 线性变换
        features = self.linear(features)  # (batch_size, embed_size)
        features = self.bn(features)
        
        return features
