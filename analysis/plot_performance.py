"""
实验一：模型架构横向对比
绘制Loss收敛图和综合雷达图
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_training_logs(log_dir):
    """加载训练日志"""
    log_file = os.path.join(log_dir, 'training_log.json')
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            return json.load(f)
    return None


def plot_loss_curves(model_logs, output_path='analysis/figures/loss_curves.png'):
    """
    绘制Loss收敛图
    Args:
        model_logs: Dict[str, Dict] 模型名称 -> 训练日志
        output_path: 输出路径
    """
    plt.figure(figsize=(12, 6))
    
    for model_name, logs in model_logs.items():
        if logs and 'train_loss' in logs:
            epochs = range(1, len(logs['train_loss']) + 1)
            plt.plot(epochs, logs['train_loss'], label=f'{model_name} (Train)', linestyle='-')
            if 'val_loss' in logs:
                plt.plot(epochs, logs['val_loss'], label=f'{model_name} (Val)', linestyle='--')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curves - Model Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Loss曲线已保存到: {output_path}")


def plot_radar_chart(model_metrics, output_path='analysis/figures/radar_chart.png'):
    """
    绘制综合雷达图
    Args:
        model_metrics: Dict[str, Dict] 模型名称 -> 指标字典
        output_path: 输出路径
    """
    # 指标列表
    metrics = ['BLEU-4', 'METEOR', 'ROUGE-L', 'CIDEr-D', 'SPICE']
    
    # 设置角度
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # 闭合
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    for model_name, scores in model_metrics.items():
        values = [scores.get(m, 0) for m in metrics]
        values += values[:1]  # 闭合
        
        # 归一化到0-1
        values = np.array(values)
        values = (values - values.min()) / (values.max() - values.min() + 1e-8)
        
        ax.plot(angles, values, 'o-', linewidth=2, label=model_name)
        ax.fill(angles, values, alpha=0.25)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.set_title('Model Performance Radar Chart', size=16, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"雷达图已保存到: {output_path}")


def main():
    """主函数"""
    # 模型配置
    models = {
        'Model A (CNN+GRU)': 'logs/cnn_gru',
        'Model B (Attn-GRU)': 'logs/attn_gru',
        'Model C (Region-Trans)': 'logs/region_transformer',
        'Model D (ViT-Trans)': 'logs/vit_transformer',
        'Model E (Graph-Trans)': 'logs/graph_transformer'
    }
    
    # 加载训练日志
    model_logs = {}
    for model_name, log_dir in models.items():
        model_logs[model_name] = load_training_logs(log_dir)
    
    # 绘制Loss曲线
    plot_loss_curves(model_logs)
    
    # 加载评估指标（需要从评估结果中读取）
    # model_metrics = {...}
    # plot_radar_chart(model_metrics)


if __name__ == '__main__':
    main()



