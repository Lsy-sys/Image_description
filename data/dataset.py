"""
DeepFashion-MultiModal数据集加载器
"""

import os
import json
import torch
import random
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class DeepFashionDataset(Dataset):
    """DeepFashion-MultiModal数据集"""
    
    def __init__(self, data_dir, split='train', transform=None, vocab=None):
        """
        Args:
            data_dir: 数据集根目录
            split: 数据集分割 ('train', 'val', 'test')
            transform: 图像变换
            vocab: 词汇表对象
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.vocab = vocab
        
        # 加载数据列表
        self.data_list = self._load_data_list()
        
    def _load_data_list(self):
        """加载数据列表"""
        list_file = os.path.join(self.data_dir, f'{self.split}_list.txt')
        with open(list_file, 'r', encoding='utf-8') as f:
            data_list = [line.strip() for line in f.readlines()]
        return data_list
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        item_id = self.data_list[idx]
        
        # 加载图像
        image_path = os.path.join(self.data_dir, 'images', f'{item_id}.jpg')
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # 加载文本描述
        text_path = os.path.join(self.data_dir, 'captions', f'{item_id}.json')
        if os.path.exists(text_path):
            with open(text_path, 'r', encoding='utf-8') as f:
                caption_data = json.load(f)
            # 获取该图片所有的参考描述 (list of strings)
            raw_captions = caption_data.get('captions', [])
        else:
            raw_captions = []

        # 随机选择一个caption进行训练 (Tensor化)
        if raw_captions:
            caption_text = random.choice(raw_captions)
            if self.vocab:
                caption_tokens = torch.tensor(self.vocab.encode(caption_text), dtype=torch.long)
            else:
                caption_tokens = torch.tensor([], dtype=torch.long)
        else:
            # 如果没有caption，返回空序列
            if self.vocab:
                caption_tokens = torch.tensor([self.vocab.word2idx['<pad>']], dtype=torch.long)
            else:
                caption_tokens = torch.tensor([], dtype=torch.long)
        
        # 返回字典，包含原始描述以便评测
        return {
            'image': image,
            'caption': caption_tokens,
            'raw_captions': raw_captions,
            'item_id': item_id
        }


class RegionDataset(Dataset):
    """区域特征数据集（用于Transformer模型）"""
    
    def __init__(self, data_dir, split='train', vocab=None):
        """
        Args:
            data_dir: 数据集根目录
            split: 数据集分割
            vocab: 词汇表对象
        """
        self.data_dir = data_dir
        self.split = split
        self.vocab = vocab
        
        # 加载数据列表
        self.data_list = self._load_data_list()
        
    def _load_data_list(self):
        """加载数据列表"""
        list_file = os.path.join(self.data_dir, f'{self.split}_list.txt')
        with open(list_file, 'r', encoding='utf-8') as f:
            data_list = [line.strip() for line in f.readlines()]
        return data_list
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        try:
            item_id = self.data_list[idx]
            
            # 1. 获取图像 (实际上Transformer模型可能需要预提取的特征，这里保留图像加载逻辑)
            # 如果你有预提取的特征，应该在这里加载 .npy 文件
            image_path = os.path.join(self.data_dir, 'images', f'{item_id}.jpg')
            if not os.path.exists(image_path):
                image = torch.zeros(3, 224, 224)
            else:
                # 这里简单处理，实际上 RegionDataset 通常配合预提取特征使用
                # 为了不报错，我们返回一个占位符或真实图像
                img = Image.open(image_path).convert('RGB')
                # 这里可以根据需要加入 transform 逻辑
            
            # 加载文本描述
            text_path = os.path.join(self.data_dir, 'captions', f'{item_id}.json')
            if os.path.exists(text_path):
                with open(text_path, 'r', encoding='utf-8') as f:
                    caption_data = json.load(f)
                raw_captions = caption_data.get('captions', [])
            else:
                raw_captions = []
            
            if raw_captions:
                caption_text = random.choice(raw_captions)
                if self.vocab:
                    caption_tokens = torch.tensor(self.vocab.encode(caption_text), dtype=torch.long)
                else:
                    caption_tokens = torch.tensor([], dtype=torch.long)
            else:
                if self.vocab:
                    caption_tokens = torch.tensor([self.vocab.word2idx['<pad>']], dtype=torch.long)
                else:
                    caption_tokens = torch.tensor([], dtype=torch.long)
            
            # 尝试加载预提取的特征
            feature_path = os.path.join(self.data_dir, 'features', f'{item_id}.npy')
            if os.path.exists(feature_path):
                regions = torch.from_numpy(np.load(feature_path)).float()
            else:
                # 没找到特征，返回零张量防止崩溃
                regions = torch.zeros(36, 2048)

            return {
                'regions': regions,
                'caption': caption_tokens,
                'raw_captions': raw_captions,
                'item_id': item_id
            }
            
        except Exception as e:
            print(f"Error loading item {idx}: {e}")
            if self.vocab:
                default_caption = torch.tensor([self.vocab.word2idx['<pad>']], dtype=torch.long)
            else:
                default_caption = torch.tensor([], dtype=torch.long)
            return {
                'regions': torch.zeros(36, 2048),
                'caption': default_caption,
                'raw_captions': [],
                'item_id': "error"
            }
