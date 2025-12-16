"""
实验四：解决"模式坍塌"的深层探究 (Decoding Strategy)
对比不同解码策略的多样性
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from typing import List, Dict


def load_strategy_results(result_dir):
    """
    加载不同策略的推理结果
    Args:
        result_dir: 结果目录，包含不同策略的JSON文件
    Returns:
        Dict[str, List] 策略名称 -> 结果列表
    """
    strategies = ['greedy', 'beam_search', 'sampling']
    results = {}
    
    for strategy in strategies:
        result_file = os.path.join(result_dir, f'predictions_{strategy}.json')
        if os.path.exists(result_file):
            with open(result_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            results[strategy] = [item.get('prediction', '') for item in data]
        else:
            print(f"警告: 未找到 {strategy} 策略的结果文件: {result_file}")
    
    return results


def compute_word_frequency(predictions: List[str]) -> Dict[str, int]:
    """
    计算词频
    Args:
        predictions: 预测句子列表
    Returns:
        词频字典
    """
    all_words = []
    for pred in predictions:
        words = pred.lower().split()
        all_words.extend(words)
    
    return Counter(all_words)


def compute_distinct_n(predictions: List[str], n: int = 2) -> float:
    """
    计算Distinct-N指标
    Args:
        predictions: 预测句子列表
        n: n-gram大小
    Returns:
        Distinct-N分数
    """
    ngrams = set()
    total_ngrams = 0
    
    for pred in predictions:
        words = pred.lower().split()
        for i in range(len(words) - n + 1):
            ngram = tuple(words[i:i+n])
            ngrams.add(ngram)
            total_ngrams += 1
    
    if total_ngrams == 0:
        return 0.0
    
    return len(ngrams) / total_ngrams


def compute_self_bleu(predictions: List[str], n: int = 4) -> float:
    """
    计算Self-BLEU（简化版）
    Args:
        predictions: 预测句子列表
        n: n-gram大小
    Returns:
        Self-BLEU分数（越低越好）
    """
    if len(predictions) < 2:
        return 0.0
    
    def get_ngrams(text, n):
        words = text.lower().split()
        return [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
    
    def compute_bleu(candidate, references):
        candidate_ngrams = get_ngrams(candidate, n)
        if not candidate_ngrams:
            return 0.0
        
        candidate_counts = Counter(candidate_ngrams)
        max_precision = 0.0
        
        for ref in references:
            ref_ngrams = get_ngrams(ref, n)
            ref_counts = Counter(ref_ngrams)
            
            overlap = sum(min(candidate_counts[ng], ref_counts[ng]) for ng in candidate_ngrams)
            precision = overlap / len(candidate_ngrams) if candidate_ngrams else 0.0
            max_precision = max(max_precision, precision)
        
        return max_precision
    
    self_bleus = []
    for i, pred in enumerate(predictions):
        references = [p for j, p in enumerate(predictions) if j != i]
        bleu = compute_bleu(pred, references)
        self_bleus.append(bleu)
    
    return np.mean(self_bleus)


def analyze_diversity(strategy_results: Dict[str, List[str]]) -> Dict[str, Dict]:
    """
    分析多样性指标
    Args:
        strategy_results: 策略结果字典
    Returns:
        分析结果
    """
    analysis = {}
    
    for strategy, predictions in strategy_results.items():
        if not predictions:
            continue
        
        # 词频
        word_freq = compute_word_frequency(predictions)
        
        # Distinct-N
        distinct_1 = compute_distinct_n(predictions, n=1)
        distinct_2 = compute_distinct_n(predictions, n=2)
        
        # Self-BLEU
        self_bleu = compute_self_bleu(predictions, n=4)
        
        analysis[strategy] = {
            'word_frequency': word_freq,
            'distinct_1': distinct_1,
            'distinct_2': distinct_2,
            'self_bleu': self_bleu,
            'vocab_size': len(word_freq),
            'total_words': sum(word_freq.values())
        }
    
    return analysis


def plot_zipf_distribution(analysis: Dict[str, Dict], output_path='analysis/figures/zipf_distribution.png'):
    """
    绘制Zipf分布图
    Args:
        analysis: 分析结果
        output_path: 输出路径
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for strategy, data in analysis.items():
        word_freq = data['word_frequency']
        
        # 按频率排序
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        frequencies = [freq for _, freq in sorted_words]
        ranks = range(1, len(frequencies) + 1)
        
        # 绘制
        label_map = {
            'greedy': 'Greedy',
            'beam_search': 'Beam Search (k=3)',
            'sampling': 'Top-k Sampling (k=5)'
        }
        label = label_map.get(strategy, strategy)
        
        ax.plot(ranks, frequencies, 'o-', label=label, linewidth=2, markersize=4, alpha=0.8)
    
    ax.set_xlabel('单词排名', fontsize=12, fontweight='bold')
    ax.set_ylabel('频率', fontsize=12, fontweight='bold')
    ax.set_title('词频 Zipf 分布图', fontsize=14, fontweight='bold', pad=20)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Zipf分布图已保存到: {output_path}")


def plot_diversity_metrics(analysis: Dict[str, Dict], output_path='analysis/figures/diversity_metrics.png'):
    """
    绘制多样性指标对比图
    Args:
        analysis: 分析结果
        output_path: 输出路径
    """
    strategies = list(analysis.keys())
    
    # 准备数据
    distinct_1_scores = [analysis[s]['distinct_1'] for s in strategies]
    distinct_2_scores = [analysis[s]['distinct_2'] for s in strategies]
    self_bleu_scores = [analysis[s]['self_bleu'] for s in strategies]
    
    # 标签映射
    label_map = {
        'greedy': 'Greedy',
        'beam_search': 'Beam Search',
        'sampling': 'Sampling'
    }
    labels = [label_map.get(s, s) for s in strategies]
    
    x = np.arange(len(strategies))
    width = 0.25
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Distinct-N对比
    ax1.bar(x - width, distinct_1_scores, width, label='Distinct-1', alpha=0.8)
    ax1.bar(x, distinct_2_scores, width, label='Distinct-2', alpha=0.8)
    ax1.set_xlabel('解码策略', fontsize=12, fontweight='bold')
    ax1.set_ylabel('分数', fontsize=12, fontweight='bold')
    ax1.set_title('Distinct-N 指标对比（越高越好）', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Self-BLEU对比
    ax2.bar(x, self_bleu_scores, width, color='coral', alpha=0.8)
    ax2.set_xlabel('解码策略', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Self-BLEU 分数', fontsize=12, fontweight='bold')
    ax2.set_title('Self-BLEU 指标对比（越低越好）', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"多样性指标对比图已保存到: {output_path}")


def print_diversity_table(analysis: Dict[str, Dict]):
    """
    打印多样性指标表格
    """
    print("\n多样性指标对比表:")
    print("=" * 80)
    print(f"{'策略':<20} {'Distinct-1':<15} {'Distinct-2':<15} {'Self-BLEU':<15} {'词汇量':<10}")
    print("-" * 80)
    
    label_map = {
        'greedy': 'Greedy',
        'beam_search': 'Beam Search',
        'sampling': 'Top-k Sampling'
    }
    
    for strategy, data in analysis.items():
        label = label_map.get(strategy, strategy)
        print(f"{label:<20} {data['distinct_1']:<15.4f} {data['distinct_2']:<15.4f} "
              f"{data['self_bleu']:<15.4f} {data['vocab_size']:<10}")
    
    print("=" * 80)
    print("\n说明:")
    print("- Distinct-N: 越高越好，表示生成的词汇多样性越高")
    print("- Self-BLEU: 越低越好，表示生成的句子重复度越低")


def main():
    """主函数"""
    # 结果目录（假设使用Model A的结果）
    result_dir = 'logs/cnn_gru'
    
    print("加载不同策略的推理结果...")
    strategy_results = load_strategy_results(result_dir)
    
    if not strategy_results:
        print("错误: 未找到任何策略的结果文件")
        print("请先使用 scripts/inference.py 生成不同策略的结果:")
        print("  python scripts/inference.py --strategy greedy --output logs/cnn_gru/predictions_greedy.json")
        print("  python scripts/inference.py --strategy beam_search --output logs/cnn_gru/predictions_beam_search.json")
        print("  python scripts/inference.py --strategy sampling --output logs/cnn_gru/predictions_sampling.json")
        return
    
    # 分析多样性
    print("\n分析多样性指标...")
    analysis = analyze_diversity(strategy_results)
    
    # 打印结果
    print_diversity_table(analysis)
    
    # 绘制图表
    print("\n绘制图表...")
    plot_zipf_distribution(analysis)
    plot_diversity_metrics(analysis)
    
    print("\n分析完成！")


if __name__ == '__main__':
    main()


