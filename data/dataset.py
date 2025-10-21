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
        with open(text_path, 'r', encoding='utf-8') as f:
            caption_data = json.load(f)
        
        # 处理多个描述
        captions = caption_data.get('captions', [])
        
        # 随机选择一个caption进行训练
        if captions:
            caption = random.choice(captions)
            if self.vocab:
                caption_tokens = torch.tensor(self.vocab.encode(caption), dtype=torch.long)
            else:
                caption_tokens = torch.tensor([], dtype=torch.long)
        else:
            # 如果没有caption，返回空序列
            if self.vocab:
                caption_tokens = torch.tensor([self.vocab.word2idx['<pad>']], dtype=torch.long)
            else:
                caption_tokens = torch.tensor([], dtype=torch.long)
        
        return image, caption_tokens


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
            
            # 加载图像
            image_path = os.path.join(self.data_dir, 'images', f'{item_id}.jpg')
            if not os.path.exists(image_path):
                print(f"Warning: Image not found: {image_path}")
                # 返回一个空白图像
                image = Image.new('RGB', (224, 224), (0, 0, 0))
            else:
                image = Image.open(image_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            # 加载文本描述
            text_path = os.path.join(self.data_dir, 'captions', f'{item_id}.json')
            if not os.path.exists(text_path):
                print(f"Warning: Caption not found: {text_path}")
                captions = []
            else:
                with open(text_path, 'r', encoding='utf-8') as f:
                    caption_data = json.load(f)
                captions = caption_data.get('captions', [])
            
            # 随机选择一个caption进行训练
            if captions:
                caption = random.choice(captions)
                if self.vocab:
                    caption_tokens = torch.tensor(self.vocab.encode(caption), dtype=torch.long)
                else:
                    caption_tokens = torch.tensor([], dtype=torch.long)
            else:
                # 如果没有caption，返回空序列
                if self.vocab:
                    caption_tokens = torch.tensor([self.vocab.word2idx['<pad>']], dtype=torch.long)
                else:
                    caption_tokens = torch.tensor([], dtype=torch.long)
            
            # 确保返回的是tuple
            result = (image, caption_tokens)
            return result
            
        except Exception as e:
            print(f"Error loading item {idx}: {e}")
            # 返回默认值
            if self.vocab:
                default_caption = torch.tensor([self.vocab.word2idx['<pad>']], dtype=torch.long)
            else:
                default_caption = torch.tensor([], dtype=torch.long)
            
            default_image = torch.zeros(3, 224, 224)
            return (default_image, default_caption)
