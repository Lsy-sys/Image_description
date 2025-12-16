"""
评测指标计算器
封装 pycocoevalcap 和其他指标
"""

from typing import List, Dict, Any
import numpy as np
from collections import defaultdict


class MetricCalculator:
    """统一的指标计算接口"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """重置所有指标"""
        self.predictions = []
        self.references = []
    
    def add_batch(self, predictions: List[str], references: List[List[str]]):
        """
        添加一批预测和参考
        
        Args:
            predictions: 预测的句子列表
            references: 参考句子列表（每个样本可以有多个参考）
        """
        self.predictions.extend(predictions)
        self.references.extend(references)
    
    def compute(self) -> Dict[str, float]:
        """
        计算所有指标
        
        Returns:
            包含各种指标的字典
        """
        from evaluation.bleu import compute_bleu
        from evaluation.cider_d import compute_cider
        from evaluation.rouge_l import compute_rouge_l
        
        # 准备参考格式（每个样本一个参考列表）
        refs = {i: refs for i, refs in enumerate(self.references)}
        preds = {i: [pred] for i, pred in enumerate(self.predictions)}
        
        # 计算BLEU
        bleu_scores = compute_bleu(refs, preds)
        
        # 计算CIDEr-D
        cider_score = compute_cider(refs, preds)
        
        # 计算ROUGE-L
        rouge_score = compute_rouge_l(refs, preds)
        
        # 计算METEOR（如果可用）
        try:
            from evaluation.meteor import compute_meteor
            meteor_score = compute_meteor(refs, preds)
        except:
            meteor_score = 0.0
        
        # 计算SPICE（如果可用）
        try:
            from evaluation.spice import compute_spice
            spice_score = compute_spice(refs, preds)
        except:
            spice_score = 0.0
        
        return {
            'BLEU-1': bleu_scores.get('Bleu_1', 0.0),
            'BLEU-2': bleu_scores.get('Bleu_2', 0.0),
            'BLEU-3': bleu_scores.get('Bleu_3', 0.0),
            'BLEU-4': bleu_scores.get('Bleu_4', 0.0),
            'CIDEr-D': cider_score,
            'ROUGE-L': rouge_score,
            'METEOR': meteor_score,
            'SPICE': spice_score
        }
    
    def compute_diversity_metrics(self) -> Dict[str, float]:
        """
        计算多样性指标（用于实验四）
        
        Returns:
            - distinct_1, distinct_2: 不同n-gram的比例
            - self_bleu: 自BLEU分数（越低越好）
        """
        # Distinct-N
        all_tokens = []
        for pred in self.predictions:
            tokens = pred.lower().split()
            all_tokens.extend(tokens)
        
        unigrams = set(all_tokens)
        bigrams = set()
        for pred in self.predictions:
            tokens = pred.lower().split()
            for i in range(len(tokens) - 1):
                bigrams.add((tokens[i], tokens[i+1]))
        
        distinct_1 = len(unigrams) / len(all_tokens) if all_tokens else 0.0
        distinct_2 = len(bigrams) / max(1, len(all_tokens) - len(self.predictions))
        
        # Self-BLEU (简化版)
        # 计算每个预测与其他所有预测的平均BLEU
        self_bleus = []
        for i, pred1 in enumerate(self.predictions):
            bleus = []
            for j, pred2 in enumerate(self.predictions):
                if i != j:
                    # 简化的BLEU计算
                    tokens1 = set(pred1.lower().split())
                    tokens2 = set(pred2.lower().split())
                    if tokens1 and tokens2:
                        overlap = len(tokens1 & tokens2)
                        bleu = overlap / len(tokens1)
                        bleus.append(bleu)
            if bleus:
                self_bleus.append(np.mean(bleus))
        
        self_bleu = np.mean(self_bleus) if self_bleus else 0.0
        
        return {
            'distinct_1': distinct_1,
            'distinct_2': distinct_2,
            'self_bleu': self_bleu
        }



