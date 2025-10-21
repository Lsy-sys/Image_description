#!/usr/bin/env python3
"""
训练CNN+GRU模型
"""

import os
import sys
import yaml
import argparse
import torch
import random
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import DeepFashionDataset
from data.transforms import ImageTransforms
from data.vocabulary import Vocabulary
from models.cnn_gru.model import CNNGruModel
from training.trainer import BaseTrainer


def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def create_directories(config):
    """创建必要的目录"""
    os.makedirs(config['paths']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['paths']['log_dir'], exist_ok=True)
    os.makedirs(os.path.dirname(config['paths']['vocab_path']), exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description='训练CNN+GRU模型')
    parser.add_argument('--config', type=str, default='configs/cnn_gru_config.yaml',
                       help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None,
                       help='恢复训练的检查点路径')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置随机种子
    set_seed(config['misc']['seed'])
    
    # 创建目录
    create_directories(config)
    
    # 创建数据变换
    train_transforms = ImageTransforms(
        image_size=config['data']['image_size'],
        is_training=True
    )
    val_transforms = ImageTransforms(
        image_size=config['data']['image_size'],
        is_training=False
    )
    
    # 创建词汇表
    vocab_path = config['paths']['vocab_path']
    if os.path.exists(vocab_path):
        print("加载现有词汇表...")
        vocab = Vocabulary()
        vocab.load(vocab_path)
    else:
        print("构建新词汇表...")
        # 这里需要先收集所有描述文本来构建词汇表
        # 简化处理，假设词汇表已存在
        vocab = Vocabulary(min_freq=config['data']['min_freq'])
        # 实际应用中需要从数据中构建词汇表
        vocab.save(vocab_path)
    
    # 创建数据集
    data_dir = config['paths']['data_dir']
    train_dataset = DeepFashionDataset(
        data_dir=data_dir,
        split='train',
        transform=train_transforms.get_transforms(),
        vocab=vocab
    )
    val_dataset = DeepFashionDataset(
        data_dir=data_dir,
        split='val',
        transform=val_transforms.get_transforms(),
        vocab=vocab
    )
    
    # 创建数据加载器
    train_loader = create_data_loader(
        dataset=train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers']
    )
    val_loader = create_data_loader(
        dataset=val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers']
    )
    
    # 创建模型
    model = CNNGruModel(
        embed_size=config['model']['embed_size'],
        hidden_size=config['model']['hidden_size'],
        vocab_size=vocab.vocab_size,
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout'],
        pretrained=config['model']['pretrained']
    )
    
    # 创建训练器
    trainer = CNNGruTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        vocab=vocab,
        config=config
    )
    
    # 恢复训练
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # 开始训练
    trainer.train()


if __name__ == '__main__':
    main()
