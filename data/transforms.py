"""
图像变换和预处理
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import random


class ImageTransforms:
    """图像变换类"""
    
    def __init__(self, image_size=224, is_training=True):
        """
        Args:
            image_size: 图像尺寸
            is_training: 是否为训练模式
        """
        self.image_size = image_size
        self.is_training = is_training
        
    def get_transforms(self):
        """获取图像变换"""
        if self.is_training:
            return self._get_training_transforms()
        else:
            return self._get_test_transforms()
    
    def _get_training_transforms(self):
        """训练时的图像变换"""
        return transforms.Compose([
            transforms.Resize((self.image_size + 32, self.image_size + 32)),
            transforms.RandomCrop(self.image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _get_test_transforms(self):
        """测试时的图像变换"""
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])


class RegionTransforms:
    """区域特征变换"""
    
    def __init__(self, max_regions=36, feature_dim=2048):
        """
        Args:
            max_regions: 最大区域数量
            feature_dim: 特征维度
        """
        self.max_regions = max_regions
        self.feature_dim = feature_dim
    
    def __call__(self, regions):
        """
        处理区域特征
        Args:
            regions: 区域特征 (num_regions, feature_dim)
        Returns:
            处理后的区域特征 (max_regions, feature_dim)
        """
        num_regions = regions.shape[0]
        
        if num_regions >= self.max_regions:
            # 随机选择max_regions个区域
            indices = random.sample(range(num_regions), self.max_regions)
            regions = regions[indices]
        else:
            # 填充到max_regions个区域
            padding = torch.zeros(self.max_regions - num_regions, self.feature_dim)
            regions = torch.cat([regions, padding], dim=0)
        
        return regions
