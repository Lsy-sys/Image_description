#!/usr/bin/env python3
"""
调试数据集加载
"""

import os
import sys
import torch
from torch.utils.data import DataLoader

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.dataset import DeepFashionDataset
from data.transforms import ImageTransforms
from data.vocabulary import Vocabulary

def collate_fn(batch):
    """自定义collate函数，处理变长序列"""
    print(f"Batch type: {type(batch)}")
    print(f"Batch length: {len(batch)}")
    print(f"First item type: {type(batch[0])}")
    print(f"First item length: {len(batch[0])}")
    
    images, captions_list = zip(*batch)
    
    # 处理图像
    images = torch.stack(images, 0)
    
    # 处理caption - 找到最大长度并padding
    max_len = max(len(caption) for caption in captions_list)
    padded_captions = []
    
    for caption in captions_list:
        if len(caption) < max_len:
            # 用0填充到最大长度
            padded = torch.cat([caption, torch.zeros(max_len - len(caption), dtype=caption.dtype)])
        else:
            padded = caption
        padded_captions.append(padded)
    
    captions = torch.stack(padded_captions, 0)
    
    return images, captions

def main():
    print("开始调试数据集...")
    
    # 创建词汇表
    print("创建词汇表...")
    vocab = Vocabulary(2)
    vocab.build_vocab_from_dataset('data/DeepFashion-MultiModal')
    
    # 创建数据集
    print("创建数据集...")
    transform = ImageTransforms(224, is_training=True)
    dataset = DeepFashionDataset('data/DeepFashion-MultiModal', 'train', transform.get_transforms(), vocab)
    
    print(f"数据集大小: {len(dataset)}")
    
    # 测试单个样本
    print("测试单个样本...")
    try:
        image, caption = dataset[0]
        print(f"Image shape: {image.shape}")
        print(f"Caption shape: {caption.shape}")
        print(f"Caption: {caption}")
        print(f"Caption type: {type(caption)}")
    except Exception as e:
        print(f"单个样本测试失败: {e}")
        return
    
    # 测试DataLoader
    print("测试DataLoader...")
    try:
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)
        for i, batch in enumerate(dataloader):
            print(f"Batch {i}:")
            print(f"  Images shape: {batch[0].shape}")
            print(f"  Captions shape: {batch[1].shape}")
            if i >= 2:  # 只测试前3个批次
                break
    except Exception as e:
        print(f"DataLoader测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
