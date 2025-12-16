"""
实验五：优化目标与奖励函数分析 (RL & Optimization)
对比XE vs RL训练，分析CIDEr vs BLEU作为奖励的效果
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, List
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import DeepFashionDataset


def load_training_logs(log_dir):
    """
    加载训练日志
    Args:
        log_dir: 日志目录
    Returns:
        训练日志数据
    """
    log_file = os.path.join(log_dir, 'training_log.json')
    if os.path.exists(log_file):
        with open(log_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def load_evaluation_results(log_dir):
    """
    加载评估结果
    Args:
        log_dir: 日志目录
    Returns:
        评估结果数据
    """
    eval_file = os.path.join(log_dir, 'evaluation_results.json')
    if os.path.exists(eval_file):
        with open(eval_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def plot_rl_training_dynamics(branches: Dict[str, Dict], output_path='analysis/figures/rl_dynamics.png'):
    """
    绘制RL训练动态折线图（The "One Chart"）
    Args:
        branches: 分支数据 {'xe': {...}, 'rl_bleu': {...}, 'rl_cider': {...}}
        output_path: 输出路径
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 准备数据
    branch_configs = {
        'xe': {'label': 'Branch 1 (XE Baseline)', 'color': 'gray', 'linestyle': '-', 'linewidth': 2},
        'rl_bleu': {'label': 'Branch 2 (RL-BLEU)', 'color': 'orange', 'linestyle': '--', 'linewidth': 2},
        'rl_cider': {'label': 'Branch 3 (RL-CIDEr)', 'color': 'blue', 'linestyle': '-', 'linewidth': 2.5}
    }
    
    for branch_name, data in branches.items():
        if branch_name not in branch_configs:
            continue
        
        config = branch_configs[branch_name]
        epochs = data.get('epochs', [])
        cider_scores = data.get('cider_scores', [])
        
        if epochs and cider_scores:
            ax.plot(epochs, cider_scores, 
                   label=config['label'],
                   color=config['color'],
                   linestyle=config['linestyle'],
                   linewidth=config['linewidth'],
                   marker='o',
                   markersize=6,
                   alpha=0.8)
    
    # 标记Epoch 20（切换点）
    ax.axvline(x=20, color='red', linestyle=':', linewidth=2, alpha=0.7, label='RL切换点 (Epoch 20)')
    ax.text(20, ax.get_ylim()[1] * 0.95, 'RL切换点', 
           rotation=90, verticalalignment='top', fontsize=10, color='red', fontweight='bold')
    
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('CIDEr Score', fontsize=12, fontweight='bold')
    ax.set_title('RL训练动态：XE vs RL-BLEU vs RL-CIDEr', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='lower right', framealpha=0.9, fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"RL训练动态图已保存到: {output_path}")


def load_reference_corpus(data_dir: str, split: str = 'train', max_samples: int = None) -> List[str]:
    """
    从数据集加载所有参考描述
    Args:
        data_dir: 数据集根目录
        split: 数据集分割 ('train', 'val', 'test')
        max_samples: 最大样本数（None表示全部）
    Returns:
        所有参考描述的列表
    """
    corpus = []
    
    # 加载数据列表
    list_file = os.path.join(data_dir, f'{split}_list.txt')
    if not os.path.exists(list_file):
        raise FileNotFoundError(f"数据列表文件不存在: {list_file}")
    
    with open(list_file, 'r', encoding='utf-8') as f:
        data_list = [line.strip() for line in f.readlines()]
    
    if max_samples:
        data_list = data_list[:max_samples]
    
    print(f"从 {split} 集加载参考描述，共 {len(data_list)} 个样本...")
    
    # 加载每个样本的所有参考描述
    captions_dir = os.path.join(data_dir, 'captions')
    for idx, item_id in enumerate(data_list):
        caption_file = os.path.join(captions_dir, f'{item_id}.json')
        
        if not os.path.exists(caption_file):
            continue
        
        try:
            with open(caption_file, 'r', encoding='utf-8') as f:
                caption_data = json.load(f)
            
            # 获取所有参考描述
            captions = caption_data.get('captions', [])
            if isinstance(captions, str):
                captions = [captions]
            
            # 添加到语料库
            for caption in captions:
                if caption and isinstance(caption, str) and len(caption.strip()) > 0:
                    corpus.append(caption.strip().lower())
        
        except Exception as e:
            if (idx + 1) % 1000 == 0:
                print(f"  处理进度: {idx + 1}/{len(data_list)}, 已加载 {len(corpus)} 条描述")
            continue
    
    print(f"成功加载 {len(corpus)} 条参考描述")
    return corpus


def compute_tfidf_weights(corpus: List[str], top_n: int = 20):
    """
    计算TF-IDF权重
    Args:
        corpus: 语料库（所有参考描述）
        top_n: 返回前N个词
    Returns:
        Dict[str, float] 词 -> TF-IDF权重
    """
    from collections import Counter
    
    # 计算词频（TF）
    all_words = []
    doc_word_counts = []
    
    for doc in corpus:
        words = doc.lower().split()
        all_words.extend(words)
        doc_word_counts.append(Counter(words))
    
    # 计算文档频率（DF）
    word_doc_count = Counter()
    for doc_words in doc_word_counts:
        word_doc_count.update(set(doc_words.keys()))
    
    total_docs = len(corpus)
    
    # 计算TF-IDF
    tfidf_weights = {}
    for word in set(all_words):
        # TF: 词在语料中的总频率
        tf = all_words.count(word) / len(all_words) if all_words else 0
        
        # IDF: 逆文档频率
        df = word_doc_count.get(word, 0)
        idf = np.log(total_docs / (df + 1)) if df > 0 else 0
        
        # TF-IDF
        tfidf_weights[word] = tf * idf
    
    # 返回前N个
    sorted_weights = sorted(tfidf_weights.items(), key=lambda x: x[1], reverse=True)
    return dict(sorted_weights[:top_n])


def plot_tfidf_analysis(corpus: List[str], output_path='analysis/figures/tfidf_analysis.png'):
    """
    绘制TF-IDF权重分析图
    Args:
        corpus: 语料库
        output_path: 输出路径
    """
    print("计算TF-IDF权重...")
    tfidf_weights = compute_tfidf_weights(corpus, top_n=30)
    
    # 分离高频词和低频词
    words = list(tfidf_weights.keys())
    weights = list(tfidf_weights.values())
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 左图：TF-IDF权重直方图
    colors = ['red' if w > np.median(weights) else 'blue' for w in weights]
    bars = ax1.barh(range(len(words)), weights, color=colors, alpha=0.7)
    ax1.set_yticks(range(len(words)))
    ax1.set_yticklabels(words, fontsize=9)
    ax1.set_xlabel('TF-IDF 权重', fontsize=12, fontweight='bold')
    ax1.set_title('TF-IDF 权重分布（前30个词）', fontsize=13, fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3, axis='x')
    
    # 添加说明
    ax1.text(0.02, 0.98, '红色：高权重（重要词）\n蓝色：低权重（常见词）',
            transform=ax1.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 右图：对比示例词
    example_words = ['floral', 'sleeveless', 'denim', 'a', 'is', 'the', 'with', 'and']
    example_weights = [tfidf_weights.get(w, 0) for w in example_words]
    
    colors_example = ['red' if w > 0.01 else 'blue' for w in example_weights]
    bars2 = ax2.bar(range(len(example_words)), example_weights, color=colors_example, alpha=0.7)
    ax2.set_xticks(range(len(example_words)))
    ax2.set_xticklabels(example_words, rotation=45, ha='right')
    ax2.set_ylabel('TF-IDF 权重', fontsize=12, fontweight='bold')
    ax2.set_title('示例词权重对比', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar, weight in zip(bars2, example_weights):
        if weight > 0:
            ax2.text(bar.get_x() + bar.get_width()/2., weight,
                   f'{weight:.4f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"TF-IDF分析图已保存到: {output_path}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='RL训练动态分析')
    parser.add_argument('--data_dir', type=str, default='data/DeepFashion-MultiModal',
                       help='数据集根目录')
    parser.add_argument('--split', type=str, default='train',
                       choices=['train', 'val', 'test'],
                       help='使用的数据集分割')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='最大样本数（None表示全部）')
    parser.add_argument('--log_dir', type=str, default='logs/region_transformer',
                       help='训练日志目录')
    
    args = parser.parse_args()
    
    # 加载三个分支的训练日志
    branches = {
        'xe': load_evaluation_results(os.path.join(args.log_dir, 'xe_branch')),
        'rl_bleu': load_evaluation_results(os.path.join(args.log_dir, 'rl_bleu_branch')),
        'rl_cider': load_evaluation_results(os.path.join(args.log_dir, 'rl_cider_branch'))
    }
    
    # 如果找不到分支结果，尝试从主日志加载
    if not any(branches.values()):
        print("尝试从主日志加载数据...")
        main_log = load_training_logs(args.log_dir)
        if main_log:
            branches = {
                'xe': {
                    'epochs': list(range(1, 31)), 
                    'cider_scores': main_log.get('xe_cider', [])
                },
                'rl_bleu': {
                    'epochs': list(range(21, 31)), 
                    'cider_scores': main_log.get('rl_bleu_cider', [])
                },
                'rl_cider': {
                    'epochs': list(range(21, 31)), 
                    'cider_scores': main_log.get('rl_cider_cider', [])
                }
            }
    
    # 绘制RL训练动态图
    if any(branches.values()):
        print("绘制RL训练动态图...")
        plot_rl_training_dynamics(branches)
    else:
        print("警告: 未找到训练日志")
        print("请先运行训练脚本生成日志文件")
        return
    
    # TF-IDF分析：从数据集真实加载参考描述
    print("\n进行TF-IDF分析...")
    print(f"从数据集加载参考描述: {args.data_dir}, split={args.split}")
    
    try:
        # 从数据集加载所有参考描述
        reference_corpus = load_reference_corpus(
            data_dir=args.data_dir,
            split=args.split,
            max_samples=args.max_samples
        )
        
        if not reference_corpus:
            raise ValueError("未能加载任何参考描述，请检查数据路径")
        
        print(f"成功加载 {len(reference_corpus)} 条参考描述用于TF-IDF分析")
        
        # 进行TF-IDF分析
        plot_tfidf_analysis(reference_corpus)
        
    except FileNotFoundError as e:
        print(f"错误: {e}")
        print("请确保数据集路径正确，并且包含 captions/ 目录和相应的JSON文件")
        return
    except Exception as e:
        print(f"加载参考描述时出错: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n分析完成！")


if __name__ == '__main__':
    main()

