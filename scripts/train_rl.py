#!/usr/bin/env python3
"""
基于强化学习的训练脚本
直接优化评测指标（BLEU、CIDEr-D、ROUGE-L等）
"""

import os
import sys
import yaml
import argparse
import torch
import random
import numpy as np
from tqdm import tqdm
import json

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import DeepFashionDataset, ImageTransforms, Vocabulary, create_data_loader
from models import CNNGruModel, TransformerModel
from training.rl_loss import RLLoss, MixedLoss


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


def load_pretrained_model(model_type, config, vocab, pretrained_path, device):
    """加载预训练模型"""
    if model_type == 'cnn_gru':
        model = CNNGruModel(
            embed_size=config['base_model']['embed_size'] if 'base_model' in config else config['model']['embed_size'],
            hidden_size=config['base_model']['hidden_size'] if 'base_model' in config else config['model']['hidden_size'],
            vocab_size=vocab.vocab_size,
            num_layers=config['base_model']['num_layers'] if 'base_model' in config else config['model']['num_layers'],
            dropout=config['base_model'].get('dropout', 0.5) if 'base_model' in config else config['model'].get('dropout', 0.5),
            pretrained=config['base_model'].get('pretrained', True) if 'base_model' in config else config['model'].get('pretrained', True)
        )
    elif model_type == 'transformer':
        model = TransformerModel(
            vocab_size=vocab.vocab_size,
            d_model=config['base_model']['d_model'],
            num_heads=config['base_model']['num_heads'],
            num_encoder_layers=config['base_model']['num_encoder_layers'],
            num_decoder_layers=config['base_model']['num_decoder_layers'],
            d_ff=config['base_model']['d_ff'],
            dropout=config['base_model']['dropout'],
            max_len=config['base_model']['max_len']
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"加载预训练模型: {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("警告: 未找到预训练模型，从头开始训练")
    
    return model


def train_epoch_rl(model, dataloader, optimizer, rl_loss, vocab, device, 
                   model_type, epoch, max_length, use_mixed_loss=False, ce_loss_weight=0.0):
    """RL训练一个epoch"""
    model.train()
    total_loss = 0
    total_reward = 0
    total_baseline = 0
    total_advantage = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f'RL Epoch {epoch}')
    
    for batch_idx, batch in enumerate(pbar):
        optimizer.zero_grad()
        
        # 准备数据
        if model_type == 'cnn_gru':
            images = batch['images'].to(device)
            raw_captions = batch['raw_captions']  # 用于计算奖励
            # 提取图像特征
            with torch.no_grad():
                image_features = model.encoder(images)
        else:  # transformer
            regions = batch['regions'].to(device)
            raw_captions = batch['raw_captions']
            # 对于transformer，regions直接作为输入
            image_features = regions
        
        # 采样序列
        sampled_seqs, log_probs = rl_loss.sample_sequences(
            model, image_features, vocab, 
            max_length=max_length,
            model_type=model_type
        )
        
        # 获取基线序列（greedy解码）
        baseline_seqs = rl_loss.get_baseline_sequences(
            model, image_features, vocab,
            max_length=max_length,
            model_type=model_type
        )
        
        # 将参考序列转换为正确的格式
        references_list = [[ref.split() for ref in refs] for refs in raw_captions]
        
        # 计算RL损失
        if use_mixed_loss and ce_loss_weight > 0:
            # 使用混合损失（需要模型的前向传播输出）
            # 这里简化处理，只使用RL损失
            # 如果需要混合损失，需要额外的实现
            raise NotImplementedError("混合损失需要额外的实现")
        else:
            loss_dict = rl_loss(
                sampled_seqs, log_probs, references_list, vocab, baseline_seqs
            )
        
        loss = loss_dict['loss']
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        # 记录统计信息
        total_loss += loss.item()
        total_reward += loss_dict.get('reward', 0)
        total_baseline += loss_dict.get('baseline', 0)
        total_advantage += loss_dict.get('advantage', 0)
        num_batches += 1
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'reward': f'{loss_dict.get("reward", 0):.4f}',
            'advantage': f'{loss_dict.get("advantage", 0):.4f}'
        })
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_reward = total_reward / num_batches if num_batches > 0 else 0
    avg_baseline = total_baseline / num_batches if num_batches > 0 else 0
    avg_advantage = total_advantage / num_batches if num_batches > 0 else 0
    
    return {
        'loss': avg_loss,
        'reward': avg_reward,
        'baseline': avg_baseline,
        'advantage': avg_advantage
    }


def validate_rl(model, dataloader, rl_loss, vocab, device, model_type, max_length):
    """验证模型"""
    model.eval()
    total_reward = 0
    num_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validating'):
            if model_type == 'cnn_gru':
                images = batch['images'].to(device)
                raw_captions = batch['raw_captions']
                image_features = model.encoder(images)
            else:
                regions = batch['regions'].to(device)
                raw_captions = batch['raw_captions']
                image_features = regions
            
            # 使用greedy解码生成描述
            baseline_seqs = rl_loss.get_baseline_sequences(
                model, image_features, vocab,
                max_length=max_length,
                model_type=model_type
            )
            
            # 计算奖励
            references_list = [[ref.split() for ref in refs] for refs in raw_captions]
            baseline_candidates = []
            for i in range(baseline_seqs.size(0)):
                seq = baseline_seqs[i].cpu().tolist()
                words = vocab.decode(seq)
                baseline_candidates.append(words.split() if words else [])
            
            rewards = rl_loss.compute_reward(baseline_candidates, references_list)
            total_reward += rewards.sum().item()
            num_samples += len(rewards)
    
    avg_reward = total_reward / num_samples if num_samples > 0 else 0
    return {'reward': avg_reward}


def main():
    parser = argparse.ArgumentParser(description='强化学习训练')
    parser.add_argument('--config', type=str, default='configs/rl_config.yaml',
                       help='配置文件路径')
    parser.add_argument('--model_type', type=str, choices=['cnn_gru', 'transformer'],
                       default='cnn_gru', help='模型类型')
    args = parser.parse_args()
    
    global config
    config = load_config(args.config)
    
    # 设置随机种子
    set_seed(config['misc'].get('seed', 42))
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建目录
    os.makedirs(config['paths']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['paths']['log_dir'], exist_ok=True)
    
    # 加载词汇表
    vocab = Vocabulary()
    if os.path.exists(config['paths']['vocab_path']):
        vocab.load(config['paths']['vocab_path'])
    else:
        print("错误: 词汇表文件不存在")
        return
    
    # 创建数据变换
    train_transforms = ImageTransforms(
        image_size=config['data'].get('image_size', 224),
        is_training=True
    )
    val_transforms = ImageTransforms(
        image_size=config['data'].get('image_size', 224),
        is_training=False
    )
    
    # 创建数据集
    data_dir = config['paths']['data_dir']
    if args.model_type == 'cnn_gru':
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
    else:
        from data import RegionDataset
        train_dataset = RegionDataset(
            data_dir=data_dir,
            split='train',
            vocab=vocab
        )
        val_dataset = RegionDataset(
            data_dir=data_dir,
            split='val',
            vocab=vocab
        )
    
    # 创建数据加载器
    train_loader = create_data_loader(
        dataset=train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data'].get('num_workers', 4)
    )
    val_loader = create_data_loader(
        dataset=val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data'].get('num_workers', 4)
    )
    
    # 加载模型
    pretrained_path = config['paths'].get('pretrained_model_path')
    model = load_pretrained_model(
        args.model_type, config, vocab, pretrained_path, device
    )
    model.to(device)
    
    # 创建优化器
    if config['training']['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training'].get('weight_decay', 0)
        )
    elif config['training']['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training'].get('weight_decay', 0)
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config['training']['learning_rate'],
            momentum=0.9
        )
    
    # 创建RL损失函数
    rl_loss = RLLoss(
        reward_type=config['rl']['reward_type'],
        baseline_type=config['rl']['baseline_type'],
        temperature=config['rl']['temperature'],
        sample_size=config['rl'].get('sample_size', 1)
    )
    
    # 训练循环
    best_reward = -float('inf')
    train_history = []
    
    num_epochs = config['training']['epochs']
    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"{'='*60}")
        
        # 训练
        max_length = config['data'].get('max_caption_length', 20)
        grad_clip = config['training'].get('grad_clip', 0)
        train_stats = train_epoch_rl(
            model, train_loader, optimizer, rl_loss, vocab, device,
            args.model_type, epoch, max_length, grad_clip
        )
        
        # 验证
        val_stats = validate_rl(
            model, val_loader, rl_loss, vocab, device, args.model_type, max_length
        )
        
        print(f"\n训练统计:")
        print(f"  损失: {train_stats['loss']:.4f}")
        print(f"  奖励: {train_stats['reward']:.4f}")
        print(f"  基线: {train_stats['baseline']:.4f}")
        print(f"  优势: {train_stats['advantage']:.4f}")
        print(f"\n验证统计:")
        print(f"  奖励: {val_stats['reward']:.4f}")
        
        # 保存最佳模型
        if val_stats['reward'] > best_reward:
            best_reward = val_stats['reward']
            checkpoint_path = os.path.join(
                config['paths']['checkpoint_dir'], 'best_model.pth'
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'reward': best_reward,
                'vocab': vocab
            }, checkpoint_path)
            print(f"\n保存最佳模型: {checkpoint_path}")
        
        # 记录历史
        train_history.append({
            'epoch': epoch,
            'train': train_stats,
            'val': val_stats
        })
        
        # 定期保存检查点
        if epoch % config['misc'].get('save_every', 5) == 0:
            checkpoint_path = os.path.join(
                config['paths']['checkpoint_dir'], f'checkpoint_epoch_{epoch}.pth'
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'reward': val_stats['reward'],
                'vocab': vocab
            }, checkpoint_path)
    
    # 保存训练历史
    history_path = os.path.join(config['paths']['log_dir'], 'training_history.json')
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(train_history, f, ensure_ascii=False, indent=2)
    print(f"\n训练历史已保存到: {history_path}")


if __name__ == '__main__':
    main()

