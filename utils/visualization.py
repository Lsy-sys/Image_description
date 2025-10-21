"""
可视化工具
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import os


def plot_training_curves(train_losses, val_losses, save_path=None):
    """
    绘制训练曲线
    Args:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        save_path: 保存路径
    """
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_metrics_comparison(metrics_dict, save_path=None):
    """
    绘制指标对比图
    Args:
        metrics_dict: 指标字典 {model_name: {metric_name: value}}
        save_path: 保存路径
    """
    models = list(metrics_dict.keys())
    metrics = list(metrics_dict[models[0]].keys())
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, model in enumerate(models):
        values = [metrics_dict[model][metric] for metric in metrics]
        ax.bar(x + i * width, values, width, label=model)
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def visualize_attention(attention_weights, image, caption, save_path=None):
    """
    可视化注意力权重
    Args:
        attention_weights: 注意力权重 (num_heads, seq_len, num_regions)
        image: 原始图像
        caption: 生成的描述
        save_path: 保存路径
    """
    num_heads = attention_weights.shape[0]
    seq_len = attention_weights.shape[1]
    num_regions = attention_weights.shape[2]
    
    # 计算平均注意力权重
    avg_attention = attention_weights.mean(axis=0)  # (seq_len, num_regions)
    
    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 显示原始图像
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # 显示生成的描述
    axes[0, 1].text(0.1, 0.5, caption, fontsize=12, wrap=True)
    axes[0, 1].set_title('Generated Caption')
    axes[0, 1].axis('off')
    
    # 显示注意力热力图
    im = axes[1, 0].imshow(avg_attention, cmap='Blues', aspect='auto')
    axes[1, 0].set_title('Attention Heatmap')
    axes[1, 0].set_xlabel('Regions')
    axes[1, 0].set_ylabel('Words')
    plt.colorbar(im, ax=axes[1, 0])
    
    # 显示每个头的注意力权重
    head_attention = attention_weights.mean(axis=1)  # (num_heads, num_regions)
    im2 = axes[1, 1].imshow(head_attention, cmap='Reds', aspect='auto')
    axes[1, 1].set_title('Attention by Head')
    axes[1, 1].set_xlabel('Regions')
    axes[1, 1].set_ylabel('Heads')
    plt.colorbar(im2, ax=axes[1, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_word_frequency(word_freq, top_k=20, save_path=None):
    """
    绘制词频分布图
    Args:
        word_freq: 词频字典
        top_k: 显示前k个词
        save_path: 保存路径
    """
    # 获取前k个高频词
    top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:top_k]
    words, freqs = zip(*top_words)
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(words)), freqs)
    plt.xticks(range(len(words)), words, rotation=45)
    plt.title(f'Top {top_k} Most Frequent Words')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_learning_rate_schedule(optimizer, epochs, save_path=None):
    """
    绘制学习率调度曲线
    Args:
        optimizer: 优化器
        epochs: 总epoch数
        save_path: 保存路径
    """
    lrs = []
    for epoch in range(epochs):
        lr = optimizer.param_groups[0]['lr']
        lrs.append(lr)
        # 这里需要根据实际的调度器调用step()
        # optimizer.step()
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(epochs), lrs)
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
