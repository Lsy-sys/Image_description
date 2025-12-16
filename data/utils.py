"""
数据处理工具函数
"""

import torch
from torch.utils.data import DataLoader
import random
from typing import Optional


def collate_fn(batch):
    """
    自定义批处理函数
    处理 Dataset 返回的字典列表
    """
    # 过滤掉可能的 None (如果在 dataset 中出错返回了 None)
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    # 检查是否是我们新的字典格式
    if isinstance(batch[0], dict):
        # 1. 处理图像 (如果有)
        if 'image' in batch[0]:
            images = torch.stack([item['image'] for item in batch])
        else:
            images = None

        # 2. 处理区域特征 (如果有)
        if 'regions' in batch[0]:
            regions = torch.stack([item['regions'] for item in batch])
        else:
            regions = None
            
        # 3. 处理描述 (训练用的 Tensor)
        captions = [item['caption'] for item in batch]
        
        # 4. 处理原始描述 (评测用的 list of strings)
        raw_captions = [item.get('raw_captions', []) for item in batch]
        
        # 5. ID
        item_ids = [item.get('item_id') for item in batch]

        return {
            'images': images,
            'regions': regions,
            'captions': captions,       # List[Tensor], 训练用
            'raw_captions': raw_captions, # List[List[str]], 评测用
            'item_ids': item_ids
        }
    
    # 兼容旧的 tuple 格式 (image, caption) - 以防万一
    elif isinstance(batch[0], tuple):
        images = torch.stack([item[0] for item in batch])
        captions = [item[1] for item in batch]

        return {
            'images': images,
            'captions': captions,
            'raw_captions': [[] for _ in batch],  # 旧格式确实没有 raw_captions
            'item_ids': [None for _ in batch],
        }
    
    else:
        raise TypeError(f"Batch format not supported: {type(batch[0])}")


def create_data_loader(dataset, batch_size=32, shuffle=True, num_workers=4, 
                      collate_fn=collate_fn):
    """
    创建数据加载器
    Args:
        dataset: 数据集
        batch_size: 批次大小
        shuffle: 是否打乱
        num_workers: 工作进程数
        collate_fn: 批处理函数
    Returns:
        数据加载器
    """
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )


def prepare_teacher_forcing_batch(batch, vocab):
    """
    准备Teacher Forcing训练批次
    Args:
        batch: 批次数据 (由 collate_fn 返回的字典)
        vocab: 词汇表
    Returns:
        输入序列和目标序列 (Tensor)
    """
    captions = batch['captions']  # List of 1D tensors
    
    # 过滤掉空描述
    valid_captions = [c for c in captions if len(c) > 0]
    if not valid_captions:
        # 如果整个batch都没描述，返回空tensor (需在训练循环中处理)
        return torch.LongTensor([]), torch.LongTensor([])

    max_length = max(len(c) for c in valid_captions)
    
    # 填充到相同长度
    input_seqs = []
    target_seqs = []
    
    pad_idx = vocab.word2idx.get('<pad>', 0)
    
    for caption in captions:
        if len(caption) == 0:
            # 处理异常空数据，填充纯PAD
            dummy = [pad_idx] * (max_length - 1)
            input_seqs.append(dummy)
            target_seqs.append(dummy)
            continue
            
        # 转为 list
        cap_list = caption.tolist()
        
        # 输入序列（去掉最后一个词 <eos>）
        input_seq = cap_list[:-1]
        # 目标序列（去掉第一个词 <sos>）
        target_seq = cap_list[1:]
        
        # 填充
        while len(input_seq) < max_length - 1:
            input_seq.append(pad_idx)
        while len(target_seq) < max_length - 1:
            target_seq.append(pad_idx)
        
        input_seqs.append(input_seq)
        target_seqs.append(target_seq)
    
    return torch.LongTensor(input_seqs), torch.LongTensor(target_seqs)
