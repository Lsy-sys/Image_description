"""
BLEU评测指标实现
支持BLEU-1、BLEU-2、BLEU-3、BLEU-4的计算
基于标准BLEU算法：基于n-gram精确率的几何平均和长度惩罚
"""

import numpy as np
import math
from collections import Counter


class BLEU:
    """BLEU评测指标类"""
    
    def __init__(self, max_n=4):
        """
        Args:
            max_n: 最大n-gram大小（默认4，即计算BLEU-1到BLEU-4）
        """
        self.max_n = max_n
        self.name = f"BLEU-{max_n}" if max_n <= 4 else f"BLEU"
    
    def get_ngrams(self, tokens, n):
        """
        获取n-gram列表
        Args:
            tokens: 词序列
            n: n-gram大小
        Returns:
            n-gram列表
        """
        if len(tokens) < n:
            return []
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    
    def compute_modified_precision(self, candidate_ngrams, reference_ngrams_list):
        """
        计算修改后的精确率（modified precision）
        这是BLEU标准算法，对每个n-gram，计算其在候选和所有参考中的匹配数
        Args:
            candidate_ngrams: 候选序列的n-gram列表
            reference_ngrams_list: 参考序列的n-gram列表的列表
        Returns:
            精确率值
        """
        if len(candidate_ngrams) == 0:
            return 0.0, 0
        
        # 统计候选序列中每个n-gram的出现次数
        candidate_counts = Counter(candidate_ngrams)
        
        # 对每个n-gram，计算在所有参考中的最大出现次数
        max_reference_counts = Counter()
        for ref_ngrams in reference_ngrams_list:
            ref_counts = Counter(ref_ngrams)
            for ngram in candidate_counts:
                max_reference_counts[ngram] = max(
                    max_reference_counts[ngram],
                    ref_counts[ngram]
                )
        
        # 计算匹配数：取候选计数和参考最大计数的最小值
        matches = 0
        for ngram in candidate_counts:
            matches += min(candidate_counts[ngram], max_reference_counts[ngram])
        
        precision = matches / len(candidate_ngrams) if len(candidate_ngrams) > 0 else 0.0
        return precision, matches
    
    def compute_brevity_penalty(self, candidate_len, reference_lens):
        """
        计算长度惩罚（Brevity Penalty, BP）
        Args:
            candidate_len: 候选序列长度
            reference_lens: 参考序列长度列表
        Returns:
            长度惩罚因子
        """
        if candidate_len == 0:
            return 0.0
        
        # 选择与候选长度最接近的参考长度
        closest_ref_len = min(reference_lens, key=lambda x: abs(x - candidate_len))
        
        if candidate_len > closest_ref_len:
            return 1.0
        else:
            # BP = exp(1 - closest_ref_len / candidate_len)
            return math.exp(1 - closest_ref_len / candidate_len)
    
    def compute_bleu_n(self, candidate, references, n):
        """
        计算BLEU-n分数
        Args:
            candidate: 候选序列（词列表）
            references: 参考序列列表（每个参考也是词列表）
            n: n-gram大小
        Returns:
            BLEU-n分数
        """
        candidate_len = len(candidate)
        reference_lens = [len(ref) for ref in references]
        
        # 如果候选序列为空，返回0
        if candidate_len == 0:
            return 0.0
        
        # 获取n-gram
        candidate_ngrams = self.get_ngrams(candidate, n)
        reference_ngrams_list = [self.get_ngrams(ref, n) for ref in references]
        
        # 计算修改后的精确率
        precision, matches = self.compute_modified_precision(
            candidate_ngrams,
            reference_ngrams_list
        )
        
        return precision
    
    def compute_bleu(self, candidate, references):
        """
        计算完整的BLEU分数（BLEU-1到BLEU-max_n的几何平均 + 长度惩罚）
        Args:
            candidate: 候选序列（词列表）
            references: 参考序列列表（每个参考也是词列表）
        Returns:
            BLEU分数
        """
        candidate_len = len(candidate)
        reference_lens = [len(ref) for ref in references]
        
        # 如果候选序列为空，返回0
        if candidate_len == 0:
            return 0.0
        
        # 计算1到max_n的精确率
        precisions = []
        for n in range(1, self.max_n + 1):
            precision = self.compute_bleu_n(candidate, references, n)
            precisions.append(precision)
        
        # 如果有任何精确率为0，直接返回0（几何平均为0）
        if any(p == 0 for p in precisions):
            return 0.0
        
        # 计算几何平均
        geometric_mean = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
        
        # 计算长度惩罚
        bp = self.compute_brevity_penalty(candidate_len, reference_lens)
        
        # 最终BLEU分数
        bleu_score = bp * geometric_mean
        
        return bleu_score
    
    def compute_bleu_1(self, candidate, references):
        """计算BLEU-1分数"""
        precision = self.compute_bleu_n(candidate, references, 1)
        candidate_len = len(candidate)
        reference_lens = [len(ref) for ref in references]
        bp = self.compute_brevity_penalty(candidate_len, reference_lens)
        return bp * precision
    
    def compute_bleu_2(self, candidate, references):
        """计算BLEU-2分数"""
        precisions = [
            self.compute_bleu_n(candidate, references, 1),
            self.compute_bleu_n(candidate, references, 2)
        ]
        if any(p == 0 for p in precisions):
            return 0.0
        geometric_mean = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
        candidate_len = len(candidate)
        reference_lens = [len(ref) for ref in references]
        bp = self.compute_brevity_penalty(candidate_len, reference_lens)
        return bp * geometric_mean
    
    def compute_bleu_3(self, candidate, references):
        """计算BLEU-3分数"""
        precisions = [
            self.compute_bleu_n(candidate, references, 1),
            self.compute_bleu_n(candidate, references, 2),
            self.compute_bleu_n(candidate, references, 3)
        ]
        if any(p == 0 for p in precisions):
            return 0.0
        geometric_mean = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
        candidate_len = len(candidate)
        reference_lens = [len(ref) for ref in references]
        bp = self.compute_brevity_penalty(candidate_len, reference_lens)
        return bp * geometric_mean
    
    def compute_bleu_4(self, candidate, references):
        """计算BLEU-4分数"""
        return self.compute_bleu(candidate, references)
    
    def compute_all_bleu(self, candidate, references):
        """
        计算所有BLEU分数（BLEU-1到BLEU-4）
        Args:
            candidate: 候选序列
            references: 参考序列列表
        Returns:
            包含BLEU-1到BLEU-4分数的字典
        """
        results = {}
        results['bleu_1'] = self.compute_bleu_1(candidate, references)
        results['bleu_2'] = self.compute_bleu_2(candidate, references)
        results['bleu_3'] = self.compute_bleu_3(candidate, references)
        results['bleu_4'] = self.compute_bleu_4(candidate, references)
        return results
    
    def compute_batch_bleu_n(self, candidates, references_list, n):
        """
        批量计算BLEU-n分数
        Args:
            candidates: 候选序列列表
            references_list: 参考序列列表的列表
        Returns:
            BLEU-n分数列表
        """
        scores = []
        for candidate, references in zip(candidates, references_list):
            score = self.compute_bleu_n(candidate, references, n)
            scores.append(score)
        return scores
    
    def compute_batch_bleu(self, candidates, references_list):
        """
        批量计算完整BLEU分数
        Args:
            candidates: 候选序列列表
            references_list: 参考序列列表的列表
        Returns:
            BLEU分数列表
        """
        scores = []
        for candidate, references in zip(candidates, references_list):
            score = self.compute_bleu(candidate, references)
            scores.append(score)
        return scores
    
    def compute_batch_all_bleu(self, candidates, references_list):
        """
        批量计算所有BLEU分数（BLEU-1到BLEU-4）
        Args:
            candidates: 候选序列列表
            references_list: 参考序列列表的列表
        Returns:
            包含BLEU-1到BLEU-4分数列表的字典
        """
        bleu_1_scores = []
        bleu_2_scores = []
        bleu_3_scores = []
        bleu_4_scores = []
        
        for candidate, references in zip(candidates, references_list):
            all_bleu = self.compute_all_bleu(candidate, references)
            bleu_1_scores.append(all_bleu['bleu_1'])
            bleu_2_scores.append(all_bleu['bleu_2'])
            bleu_3_scores.append(all_bleu['bleu_3'])
            bleu_4_scores.append(all_bleu['bleu_4'])
        
        return {
            'bleu_1': bleu_1_scores,
            'bleu_2': bleu_2_scores,
            'bleu_3': bleu_3_scores,
            'bleu_4': bleu_4_scores
        }

