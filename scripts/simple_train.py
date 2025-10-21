#!/usr/bin/env python3
"""
简化的CNN+GRU模型训练脚本
"""

import os
import sys
import yaml
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import numpy as np
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import DeepFashionDataset
from data.transforms import ImageTransforms
from data.vocabulary import Vocabulary
from models.cnn_gru.model import CNNGruModel

def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def collate_fn(batch):
    """自定义collate函数，处理变长序列"""
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

def train_epoch(model, dataloader, optimizer, criterion, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    
    for batch_idx, (images, captions) in enumerate(tqdm(dataloader, desc="Training")):
        images, captions = images.to(device), captions.to(device)
        
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(images, captions[:, :-1])  # 输入不包括最后一个token
        
        # 计算损失
        loss = criterion(outputs.reshape(-1, outputs.shape[-1]), captions[:, 1:].reshape(-1))
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
    
    return total_loss / len(dataloader)

def validate_epoch(model, dataloader, criterion, device):
    """验证一个epoch"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for images, captions in tqdm(dataloader, desc="Validation"):
            images, captions = images.to(device), captions.to(device)
            
            outputs = model(images, captions[:, :-1])
            loss = criterion(outputs.reshape(-1, outputs.shape[-1]), captions[:, 1:].reshape(-1))
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def main():
    parser = argparse.ArgumentParser(description='训练CNN+GRU模型')
    parser.add_argument('--config', type=str, default='configs/cnn_gru_config.yaml',
                       help='配置文件路径')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=None,
                       help='学习率')
    parser.add_argument('--epochs', type=int, default=None,
                       help='训练轮数')
    parser.add_argument('--device', type=str, default='auto',
                       help='设备 (auto, cuda, cpu)')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 覆盖命令行参数
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate
    if args.epochs:
        config['training']['epochs'] = args.epochs
    
    # 设置设备
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"使用设备: {device}")
    
    # 设置随机种子
    set_seed(config['misc']['seed'])
    
    # 创建目录
    create_directories(config)
    
    # 创建词汇表
    print("创建词汇表...")
    vocab = Vocabulary(config['data']['min_freq'])
    
    # 构建词汇表
    print("构建词汇表...")
    vocab.build_vocab_from_dataset(config['paths']['data_dir'])
    
    # 加载数据集
    print("加载数据集...")
    train_transform = ImageTransforms(config['data']['image_size'], is_training=True)
    val_transform = ImageTransforms(config['data']['image_size'], is_training=False)
    
    train_dataset = DeepFashionDataset(
        config['paths']['data_dir'], 
        split='train', 
        transform=train_transform.get_transforms(),
        vocab=vocab
    )
    
    val_dataset = DeepFashionDataset(
        config['paths']['data_dir'], 
        split='val', 
        transform=val_transform.get_transforms(),
        vocab=vocab
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=True,
        num_workers=config['data']['num_workers'],
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=False,
        num_workers=config['data']['num_workers'],
        collate_fn=collate_fn
    )
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"词汇表大小: {len(vocab)}")
    
    # 创建模型
    print("创建模型...")
    model = CNNGruModel(
        embed_size=config['model']['embed_size'],
        hidden_size=config['model']['hidden_size'],
        vocab_size=len(vocab),
        num_layers=config['model']['num_layers'],
        pretrained=config['model']['pretrained']
    ).to(device)
    
    # 创建优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略padding token
    
    # 训练循环
    print("开始训练...")
    best_val_loss = float('inf')
    
    for epoch in range(config['training']['epochs']):
        print(f"\nEpoch {epoch+1}/{config['training']['epochs']}")
        
        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # 验证
        val_loss = validate_epoch(model, val_loader, criterion, device)
        
        print(f"训练损失: {train_loss:.4f}")
        print(f"验证损失: {val_loss:.4f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'vocab': vocab
            }, os.path.join(config['paths']['checkpoint_dir'], 'best_model.pth'))
            print("保存最佳模型")
        
        # 定期保存检查点
        if (epoch + 1) % config['misc']['save_every'] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'vocab': vocab
            }, os.path.join(config['paths']['checkpoint_dir'], f'checkpoint_epoch_{epoch+1}.pth'))
    
    print("训练完成！")

if __name__ == '__main__':
    main()
