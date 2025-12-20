#!/usr/bin/env python3
"""
统一的训练入口
支持所有5个模型的训练
"""

import os
import sys
import argparse
import yaml
import torch
import random
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import create_model_from_config, load_config
from data.dataset import DeepFashionDataset
from data.graph_dataset import GraphDataset
from data.vocabulary import Vocabulary
from data.transforms import ImageTransforms
from modules import CrossEntropyLoss, create_optimizer, create_scheduler, MetricCalculator
from training.trainer import BaseTrainer


def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description='统一训练入口')
    parser.add_argument('--config', type=str, required=True,
                       help='模型配置文件路径（如 configs/models/1_cnn_gru.yaml）')
    parser.add_argument('--strategy', type=str, default='xe_training',
                       help='训练策略配置（xe_training 或 rl_finetune）')
    parser.add_argument('--resume', type=str, default=None,
                       help='恢复训练的检查点路径')
    parser.add_argument('--device', type=str, default='auto',
                       help='运行设备 (auto/cuda/cpu)')
    
    args = parser.parse_args()
    
    # 设置随机种子
    config = load_config(args.config)
    seed = config.get('misc', {}).get('seed', 42)
    set_seed(seed)
    
    # 确定设备
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"使用设备: {device}")
    
    # 加载词汇表（优先从文件加载，不存在则构建）
    vocab_path = config['paths'].get('vocab_path', 'data/vocab.json')
    if os.path.exists(vocab_path):
        print(f"从文件加载词汇表: {vocab_path}")
        vocab = Vocabulary.load(vocab_path)
        print(f"词汇表大小: {len(vocab)} (min_freq={vocab.min_freq})")
        # 检查 min_freq 是否匹配（如果不匹配，给出警告）
        config_min_freq = config['data'].get('min_freq', 5)
        if vocab.min_freq != config_min_freq:
            print(f"警告: 配置中的 min_freq={config_min_freq} 与已加载词表的 min_freq={vocab.min_freq} 不一致")
            print(f"      如需重新构建词表，请删除 {vocab_path} 后重新训练")
    else:
        print(f"词汇表文件不存在，从数据集构建...")
        vocab = Vocabulary.from_captions(
            config['paths']['data_dir'],
            min_freq=config['data']['min_freq'],
            vocab_path=vocab_path
        )
        print(f"词汇表已构建并保存到: {vocab_path}")
    
    vocab_size = len(vocab)
    
    # 创建数据集
    print("创建数据集...")
    model_type = config['model']['type']
    
    if model_type == 'graph_transformer':
        # Graph-Transformer 使用 GraphDataset
        feature_dir = config['paths'].get('feature_dir', 'data/features')
        graph_dir = config['paths'].get('graph_dir', 'data/graphs')
        default_num_nodes = config['model']['encoder'].get('num_nodes', 196)
        default_node_feature_dim = config['model']['encoder'].get('node_feature_dim', 768)
        train_dataset = GraphDataset(
            data_dir=config['paths']['data_dir'],
            feature_dir=feature_dir,
            graph_dir=graph_dir,
            split='train',
            vocab=vocab,
            default_num_nodes=default_num_nodes,
            default_node_feature_dim=default_node_feature_dim,
        )
        val_dataset = GraphDataset(
            data_dir=config['paths']['data_dir'],
            feature_dir=feature_dir,
            graph_dir=graph_dir,
            split='val',
            vocab=vocab,
            default_num_nodes=default_num_nodes,
            default_node_feature_dim=default_node_feature_dim,
        )
    else:
        # 其他模型使用 DeepFashionDataset
        transform = ImageTransforms(config['data']['image_size'], is_training=True)
        train_dataset = DeepFashionDataset(
            config['paths']['data_dir'],
            split='train',
            transform=transform.get_transforms(),
            vocab=vocab
        )
        val_dataset = DeepFashionDataset(
            config['paths']['data_dir'],
            split='val',
            transform=transform.get_transforms(),
            vocab=vocab
        )
    
    # 创建模型
    print("创建模型...")
    model = create_model_from_config(args.config, vocab_size)
    model = model.to(device)
    
    # 加载训练策略配置
    strategy_config_path = f"configs/strategies/{args.strategy}.yaml"
    if os.path.exists(strategy_config_path):
        strategy_config = load_config(strategy_config_path)
    else:
        strategy_config = {}
    
    # 创建损失函数
    if args.strategy == 'rl_finetune':
        from modules import SCSTLoss
        loss_fn = SCSTLoss(
            reward_type=strategy_config.get('loss', {}).get('reward_type', 'cider_d'),
            baseline_type=strategy_config.get('loss', {}).get('baseline_type', 'self_critical'),
            temperature=strategy_config.get('loss', {}).get('temperature', 1.0)
        )
    else:
        loss_fn = CrossEntropyLoss(
            ignore_index=vocab.word2idx.get('<pad>', 0),
            label_smoothing=strategy_config.get('loss', {}).get('label_smoothing', 0.0)
        )
    
    # 创建优化器和调度器
    optimizer = create_optimizer(model, strategy_config.get('optimizer', config['training']))
    scheduler = create_scheduler(
        optimizer,
        strategy_config.get('scheduler', {}),
        num_training_steps=len(train_dataset) // config['training']['batch_size'] * config['training']['epochs']
    )
    
    # 创建训练器
    trainer = BaseTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config,
        vocab=vocab
    )
    
    # 恢复训练
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # 开始训练
    print("开始训练...")
    trainer.train()
    
    print("训练完成！")


if __name__ == '__main__':
    main()






