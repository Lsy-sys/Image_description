"""
指标计算工具
"""

import numpy as np


class AverageMeter:
    """计算平均值的工具类"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """重置"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        """更新"""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """
    计算top-k准确率
    Args:
        output: 模型输出 (batch_size, num_classes)
        target: 目标标签 (batch_size,)
        topk: 要计算的top-k值
    Returns:
        top-k准确率列表
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        
        return res


def bleu_score(candidate, references, n=4):
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
