#!/usr/bin/env python3
"""
简化版Transformer模型训练脚本
避免NumPy版本冲突问题
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import json
import time

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.transformer.model import TransformerModel
from data.dataset import DeepFashionDataset
from data.vocabulary import Vocabulary
from data.transforms import ImageTransforms

def collate_fn(batch):
    """自定义collate函数，处理不同长度的序列"""
    images, captions = zip(*batch)
    
    # 堆叠图像
    images = torch.stack(images, 0)
    
    # 计算序列长度
    lengths = [len(caption) for caption in captions]
    
    # 找到最大序列长度
    max_len = max(lengths)
    
    # 填充序列到相同长度
    padded_captions = []
    for caption in captions:
        if len(caption) < max_len:
            # 使用PAD标记填充
            pad_length = max_len - len(caption)
            padded_caption = torch.cat([caption, torch.zeros(pad_length, dtype=caption.dtype)])
        else:
            padded_caption = caption
        padded_captions.append(padded_caption)
    
    captions = torch.stack(padded_captions, 0)
    lengths = torch.tensor(lengths)
    
    return images, captions, lengths

class RegionFeatureExtractor:
    """区域特征提取器（训练版本）"""
    
    def __init__(self, device):
        self.device = device
        # 使用ResNet-50作为特征提取器
        import torchvision.models as models
        self.backbone = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        self.backbone.eval()
        self.backbone.to(device)
        
    def extract_regions(self, image_tensor, num_regions=36):
        """提取区域特征"""
        with torch.no_grad():
            # 提取全局特征
            global_features = self.backbone(image_tensor)
            global_features = global_features.squeeze(-1).squeeze(-1)
            
            # 复制全局特征作为多个区域特征
            region_features = global_features.unsqueeze(1).repeat(1, num_regions, 1)
            
            # 添加随机噪声
            noise = torch.randn_like(region_features) * 0.1
            region_features = region_features + noise
            
            return region_features

class SimpleLogger:
    """简化的日志记录器"""
    
    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, 'training.log')
        
    def log(self, data):
        """记录训练数据"""
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] " + " | ".join([f"{k}: {v:.4f}" for k, v in data.items()])
        
        print(log_entry)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry + '\n')

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def create_model(config, vocab_size):
    """创建Transformer模型"""
    model = TransformerModel(
        vocab_size=vocab_size,
        d_model=config['model']['d_model'],
        num_heads=config['model']['num_heads'],
        num_encoder_layers=config['model']['num_encoder_layers'],
        num_decoder_layers=config['model']['num_decoder_layers'],
        d_ff=config['model']['d_ff'],
        dropout=config['model']['dropout'],
        max_len=config['model']['max_len']
    )
    return model

def train_epoch(model, dataloader, optimizer, criterion, device, region_extractor, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, (images, captions, lengths) in enumerate(pbar):
        images = images.to(device)
        captions = captions.to(device)
        
        # 提取区域特征
        regions = region_extractor.extract_regions(images)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(regions, captions)
        
        # 计算损失
        loss = criterion(outputs.view(-1, outputs.size(-1)), captions.view(-1))
        
        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(dataloader)

def validate(model, dataloader, device, region_extractor, vocab):
    """验证模型"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_references = []
    
    with torch.no_grad():
        for images, captions, lengths in tqdm(dataloader, desc='Validating'):
            images = images.to(device)
            captions = captions.to(device)
            
            # 提取区域特征
            regions = region_extractor.extract_regions(images)
            
            # 前向传播
            outputs = model(regions, captions)
            
            # 计算损失
            criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)
            loss = criterion(outputs.view(-1, outputs.size(-1)), captions.view(-1))
            total_loss += loss.item()
            
            # 生成描述用于评估
            generated = model.generate(regions, vocab, max_length=20)
            
            # 转换为文本
            for i in range(generated.size(0)):
                pred_caption = []
                for word_id in generated[i]:
                    word = vocab.idx2word[word_id.item()]
                    if word == vocab.EOS_TOKEN:
                        break
                    if word not in [vocab.SOS_TOKEN, vocab.PAD_TOKEN, vocab.UNK_TOKEN]:
                        pred_caption.append(word)
                
                all_predictions.append(' '.join(pred_caption))
                
                # 获取参考描述
                ref_caption = []
                for word_id in captions[i]:
                    word = vocab.idx2word[word_id.item()]
                    if word == vocab.EOS_TOKEN:
                        break
                    if word not in [vocab.SOS_TOKEN, vocab.PAD_TOKEN, vocab.UNK_TOKEN]:
                        ref_caption.append(word)
                
                all_references.append([' '.join(ref_caption)])
    
    # 简化的评估（不使用可能有问题的评估模块）
    avg_loss = total_loss / len(dataloader)
    
    # 简单的BLEU-like评估
    total_matches = 0
    total_words = 0
    
    for pred, ref_list in zip(all_predictions, all_references):
        ref = ref_list[0] if ref_list else ""
        pred_words = pred.split()
        ref_words = ref.split()
        
        # 计算词级别的匹配
        matches = sum(1 for word in pred_words if word in ref_words)
        total_matches += matches
        total_words += len(pred_words)
    
    bleu_score = total_matches / total_words if total_words > 0 else 0.0
    
    return avg_loss, bleu_score

def main():
    parser = argparse.ArgumentParser(description='简化版Transformer模型训练')
    parser.add_argument('--config', required=True, help='配置文件路径')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='学习率')
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'], help='运行设备')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 设备选择
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"使用设备: {device}")
    
    # 创建目录
    os.makedirs(config['paths']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['paths']['log_dir'], exist_ok=True)
    
    # 创建词汇表
    print("构建词汇表...")
    vocab = Vocabulary(min_freq=config['data']['min_freq'])
    vocab.build_vocab_from_dataset(config['paths']['data_dir'])
    
    # 创建数据变换
    image_transform = ImageTransforms(224, is_training=True)
    
    # 创建数据集
    print("创建数据集...")
    train_dataset = DeepFashionDataset(
        data_dir=config['paths']['data_dir'],
        split='train',
        vocab=vocab,
        transform=image_transform.get_transforms()
    )
    
    val_dataset = DeepFashionDataset(
        data_dir=config['paths']['data_dir'],
        split='val',
        vocab=vocab,
        transform=image_transform.get_transforms()
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,  # 减少worker数量避免问题
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn
    )
    
    # 创建模型
    print("创建Transformer模型...")
    model = create_model(config, len(vocab))
    model.to(device)
    
    # 创建区域特征提取器
    region_extractor = RegionFeatureExtractor(device)
    
    # 创建优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=config['training']['weight_decay']
    )
    
    # 创建损失函数
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)
    
    # 创建日志记录器
    logger = SimpleLogger(config['paths']['log_dir'])
    
    # 训练循环
    best_score = 0
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, region_extractor, epoch+1)
        
        # 验证
        val_loss, bleu_score = validate(model, val_loader, device, region_extractor, vocab)
        
        # 记录日志
        logger.log({
            'epoch': epoch+1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'bleu_score': bleu_score
        })
        
        print(f"训练损失: {train_loss:.4f}")
        print(f"验证损失: {val_loss:.4f}")
        print(f"BLEU分数: {bleu_score:.4f}")
        
        # 保存最佳模型
        if bleu_score > best_score:
            best_score = bleu_score
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'vocab': vocab,
                'config': config,
                'score': bleu_score
            }, os.path.join(config['paths']['checkpoint_dir'], 'best_model.pth'))
            print(f"保存最佳模型，得分: {bleu_score:.4f}")
        
        # 定期保存检查点
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'vocab': vocab,
                'config': config,
                'score': bleu_score
            }, os.path.join(config['paths']['checkpoint_dir'], f'checkpoint_epoch_{epoch+1}.pth'))
    
    print("训练完成！")

if __name__ == '__main__':
    main()
