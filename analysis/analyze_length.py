"""
实验三：序列长度敏感度分析 (Robustness)
验证模型生成长文本（Long Caption）的能力
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path


def load_inference_results(result_dir):
    """
    加载推理结果
    Args:
        result_dir: 结果目录，包含 predictions.json
    Returns:
        List of (image_id, prediction, reference, length)
    """
    result_file = os.path.join(result_dir, 'predictions.json')
    if not os.path.exists(result_file):
        print(f"警告: 结果文件不存在: {result_file}")
        return []
    
    with open(result_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = []
    for item in data:
        image_id = item.get('image_id', '')
        prediction = item.get('prediction', '')
        references = item.get('references', [])
        
        # 计算参考长度（取平均）
        if references:
            ref_lengths = [len(ref.split()) for ref in references]
            avg_length = np.mean(ref_lengths)
        else:
            avg_length = len(prediction.split())
        
        results.append({
            'image_id': image_id,
            'prediction': prediction,
            'references': references,
            'length': avg_length
        })
    
    return results


def categorize_by_length(results, thresholds=(15, 25)):
    """
    根据长度分类
    Args:
        results: 推理结果列表
        thresholds: (short_max, medium_max) 阈值
    Returns:
        Dict[str, List] 分类后的结果
    """
    short_max, medium_max = thresholds
    
    categorized = {
        'Short (<15词)': [],
        'Medium (15-25词)': [],
        'Long (>25词)': []
    }
    
    for item in results:
        length = item['length']
        if length < short_max:
            categorized['Short (<15词)'].append(item)
        elif length < medium_max:
            categorized['Medium (15-25词)'].append(item)
        else:
            categorized['Long (>25词)'].append(item)
    
    return categorized


def compute_cider_score(predictions, references):
    """
    计算CIDEr分数（简化版）
    实际应该使用evaluation/cider_d.py
    """
    try:
        from evaluation.cider_d import compute_cider
        
        # 准备格式
        refs = {i: [ref] for i, ref in enumerate(references)}
        preds = {i: [pred] for i, pred in enumerate(predictions)}
        
        score = compute_cider(refs, preds)
        return score
    except:
        # 如果无法导入，返回0
        return 0.0


def analyze_length_sensitivity(model_results):
    """
    分析长度敏感度
    Args:
        model_results: Dict[str, List] 模型名称 -> 结果列表
    Returns:
        Dict[str, Dict] 分析结果
    """
    analysis = {}
    
    for model_name, results in model_results.items():
        # 分类
        categorized = categorize_by_length(results)
        
        # 计算每个类别的平均CIDEr
        category_scores = {}
        for category, items in categorized.items():
            if not items:
                category_scores[category] = 0.0
                continue
            
            predictions = [item['prediction'] for item in items]
            all_references = [item['references'] for item in items]
            
            # 计算CIDEr（简化处理）
            scores = []
            for pred, refs in zip(predictions, all_references):
                if refs:
                    score = compute_cider_score([pred], refs)
                    scores.append(score)
            
            category_scores[category] = np.mean(scores) if scores else 0.0
        
        analysis[model_name] = category_scores
    
    return analysis


def plot_length_sensitivity(analysis, output_path='analysis/figures/length_sensitivity.png'):
    """
    绘制长度敏感度柱状图
    Args:
        analysis: 分析结果
        output_path: 输出路径
    """
    categories = ['Short (<15词)', 'Medium (15-25词)', 'Long (>25词)']
    models = list(analysis.keys())
    
    # 准备数据
    x = np.arange(len(categories))
    width = 0.15
    spacing = 0.05
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 为每个模型绘制柱状图
    for i, model_name in enumerate(models):
        scores = [analysis[model_name].get(cat, 0.0) for cat in categories]
        offset = (i - len(models) / 2) * (width + spacing)
        bars = ax.bar(x + offset, scores, width, label=model_name, alpha=0.8)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('序列长度区间', fontsize=12, fontweight='bold')
    ax.set_ylabel('平均 CIDEr 分数', fontsize=12, fontweight='bold')
    ax.set_title('序列长度敏感度分析', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"长度敏感度分析图已保存到: {output_path}")


def main():
    """主函数"""
    # 模型结果目录配置
    model_dirs = {
        'Model A (CNN+GRU)': 'logs/cnn_gru',
        'Model B (Attn-GRU)': 'logs/attn_gru',
        'Model C (Region-Trans)': 'logs/region_transformer',
        'Model D (ViT-Trans)': 'logs/vit_transformer',
        'Model E (Graph-Trans)': 'logs/graph_transformer'
    }
    
    # 加载所有模型的结果
    model_results = {}
    for model_name, result_dir in model_dirs.items():
        print(f"加载 {model_name} 的结果...")
        results = load_inference_results(result_dir)
        if results:
            model_results[model_name] = results
            print(f"  - 加载了 {len(results)} 个样本")
        else:
            print(f"  - 警告: 未找到结果文件")
    
    if not model_results:
        print("错误: 未找到任何模型结果，请先运行推理脚本生成结果")
        return
    
    # 分析长度敏感度
    print("\n分析长度敏感度...")
    analysis = analyze_length_sensitivity(model_results)
    
    # 打印结果
    print("\n长度敏感度分析结果:")
    print("-" * 60)
    for model_name, scores in analysis.items():
        print(f"\n{model_name}:")
        for category, score in scores.items():
            print(f"  {category}: {score:.4f}")
    
    # 绘制图表
    print("\n绘制图表...")
    plot_length_sensitivity(analysis)
    
    print("\n分析完成！")


if __name__ == '__main__':
    main()
















