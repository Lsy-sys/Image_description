"""
评测工具函数
"""

import numpy as np
from .rouge_l import RougeL
from .cider_d import CiderD


def compute_metrics(candidates, references_list, metrics=['rouge_l', 'cider_d']):
    """
    计算多种评测指标
    Args:
        candidates: 候选序列列表
        references_list: 参考序列列表的列表
        metrics: 要计算的指标列表
    Returns:
        指标分数字典
    """
    results = {}
    
    if 'rouge_l' in metrics:
        rouge_l = RougeL()
        rouge_scores = rouge_l.compute_batch_rouge_l(candidates, references_list)
        results['rouge_l'] = {
            'scores': rouge_scores,
            'mean': np.mean(rouge_scores),
            'std': np.std(rouge_scores)
        }
    
    if 'cider_d' in metrics:
        cider_d = CiderD()
        cider_scores = cider_d.compute_batch_cider_d(candidates, references_list)
        results['cider_d'] = {
            'scores': cider_scores,
            'mean': np.mean(cider_scores),
            'std': np.std(cider_scores)
        }
    
    return results


def compute_bleu_score(candidate, references, n=4):
    """
    计算BLEU分数
    Args:
        candidate: 候选序列
        references: 参考序列列表
        n: n-gram大小
    Returns:
        BLEU分数
    """
    from collections import Counter
    
    def get_ngrams(seq, n):
        return [tuple(seq[i:i+n]) for i in range(len(seq)-n+1)]
    
    def compute_precision(candidate_ngrams, reference_ngrams):
        if len(candidate_ngrams) == 0:
            return 0.0
        
        candidate_counts = Counter(candidate_ngrams)
        reference_counts = Counter(reference_ngrams)
        
        overlap = 0
        for ngram in candidate_counts:
            overlap += min(candidate_counts[ngram], reference_counts[ngram])
        
        return overlap / len(candidate_ngrams)
    
    # 计算1到n-gram的精确率
    precisions = []
    for i in range(1, n + 1):
        candidate_ngrams = get_ngrams(candidate, i)
        reference_ngrams = [get_ngrams(ref, i) for ref in references]
        
        # 计算与每个参考的最大精确率
        max_precision = 0
        for ref_ngrams in reference_ngrams:
            precision = compute_precision(candidate_ngrams, ref_ngrams)
            max_precision = max(max_precision, precision)
        
        precisions.append(max_precision)
    
    # 计算几何平均
    if any(p == 0 for p in precisions):
        return 0.0
    
    bleu = np.exp(np.mean(np.log(precisions)))
    
    # 长度惩罚
    candidate_len = len(candidate)
    reference_lens = [len(ref) for ref in references]
    closest_ref_len = min(reference_lens, key=lambda x: abs(x - candidate_len))
    
    if candidate_len < closest_ref_len:
        bp = np.exp(1 - closest_ref_len / candidate_len)
    else:
        bp = 1.0
    
    return bp * bleu


def compute_meteor_score(candidate, references):
    """
    计算METEOR分数
    Args:
        candidate: 候选序列
        references: 参考序列列表
    Returns:
        METEOR分数
    """
    # 简化的METEOR实现
    # 实际应用中可以使用NLTK的METEOR实现
    
    def get_unigrams(seq):
        return seq
    
    def get_bigrams(seq):
        return [tuple(seq[i:i+2]) for i in range(len(seq)-1)]
    
    def compute_matches(candidate_grams, reference_grams):
        candidate_counts = Counter(candidate_grams)
        reference_counts = Counter(reference_grams)
        
        matches = 0
        for gram in candidate_counts:
            matches += min(candidate_counts[gram], reference_counts[gram])
        
        return matches
    
    # 计算unigram和bigram匹配
    candidate_unigrams = get_unigrams(candidate)
    candidate_bigrams = get_bigrams(candidate)
    
    max_matches = 0
    for ref in references:
        ref_unigrams = get_unigrams(ref)
        ref_bigrams = get_bigrams(ref)
        
        unigram_matches = compute_matches(candidate_unigrams, ref_unigrams)
        bigram_matches = compute_matches(candidate_bigrams, ref_bigrams)
        
        # 加权匹配数
        matches = unigram_matches + 0.5 * bigram_matches
        max_matches = max(max_matches, matches)
    
    if len(candidate) == 0:
        return 0.0
    
    precision = max_matches / len(candidate)
    recall = max_matches / sum(len(ref) for ref in references) * len(references)
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * precision * recall / (precision + recall)
    
    # 长度惩罚
    candidate_len = len(candidate)
    reference_lens = [len(ref) for ref in references]
    closest_ref_len = min(reference_lens, key=lambda x: abs(x - candidate_len))
    
    if candidate_len < closest_ref_len:
        penalty = 0.5 * (candidate_len / closest_ref_len) ** 3
    else:
        penalty = 1.0
    
    return f1 * penalty
