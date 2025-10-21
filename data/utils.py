"""
数据处理工具函数
"""

import torch
from torch.utils.data import DataLoader
import random


def collate_fn(batch):
    """
    自定义批处理函数
    Args:
        batch: 批次数据
    Returns:
        处理后的批次数据
    """
    images = torch.stack([item['image'] for item in batch])
    
    # 处理多个描述
    captions = []
    raw_captions = []
    item_ids = []
    
    for item in batch:
        # 随机选择一个描述
        if item['captions']:
            caption = random.choice(item['captions'])
            captions.append(torch.LongTensor(caption))
        else:
            captions.append(torch.LongTensor([]))
        
        raw_captions.append(item['raw_captions'])
        item_ids.append(item['item_id'])
    
    return {
        'images': images,
        'captions': captions,
        'raw_captions': raw_captions,
        'item_ids': item_ids
    }


def region_collate_fn(batch):
    """
    区域特征批处理函数
    Args:
        batch: 批次数据
    Returns:
        处理后的批次数据
    """
    regions = torch.stack([item['regions'] for item in batch])
    
    # 处理多个描述
    captions = []
    raw_captions = []
    item_ids = []
    
    for item in batch:
        if item['captions']:
            caption = random.choice(item['captions'])
            captions.append(torch.LongTensor(caption))
        else:
            captions.append(torch.LongTensor([]))
        
        raw_captions.append(item['raw_captions'])
        item_ids.append(item['item_id'])
    
    return {
        'regions': regions,
        'captions': captions,
        'raw_captions': raw_captions,
        'item_ids': item_ids
    }


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


def create_vocabulary_from_captions(captions_list, min_freq=5, save_path=None):
    """
    从描述文本创建词汇表
    Args:
        captions_list: 所有描述文本列表
        min_freq: 最小词频
        save_path: 保存路径
    Returns:
        词汇表对象
    """
    from .vocabulary import Vocabulary
    
    vocab = Vocabulary(min_freq=min_freq)
    vocab.build_vocab(captions_list)
    
    if save_path:
        vocab.save(save_path)
    
    return vocab


def prepare_teacher_forcing_batch(batch, vocab):
    """
    准备Teacher Forcing训练批次
    Args:
        batch: 批次数据
        vocab: 词汇表
    Returns:
        输入序列和目标序列
    """
    captions = batch['captions']
    max_length = max(len(caption) for caption in captions)
    
    # 填充到相同长度
    input_seqs = []
    target_seqs = []
    
    for caption in captions:
        if len(caption) == 0:
            continue
            
        # 输入序列（去掉最后一个词）
        input_seq = caption[:-1].tolist()
        # 目标序列（去掉第一个词）
        target_seq = caption[1:].tolist()
        
        # 填充
        while len(input_seq) < max_length - 1:
            input_seq.append(vocab.pad_idx)
        while len(target_seq) < max_length - 1:
            target_seq.append(vocab.pad_idx)
        
        input_seqs.append(input_seq)
        target_seqs.append(target_seq)
    
    return torch.LongTensor(input_seqs), torch.LongTensor(target_seqs)
