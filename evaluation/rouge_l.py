"""
ROUGE-L评测指标实现
基于最长公共子序列(LCS)
"""

import numpy as np
from collections import Counter


class RougeL:
    """ROUGE-L评测指标"""
    
    def __init__(self):
        self.name = "ROUGE-L"
    
    def compute_lcs(self, seq1, seq2):
        """
        计算两个序列的最长公共子序列长度
        Args:
            seq1: 序列1
            seq2: 序列2
        Returns:
            LCS长度
        """
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    def compute_rouge_l(self, candidate, reference):
        """
        计算ROUGE-L分数
        Args:
            candidate: 候选序列
            reference: 参考序列
        Returns:
            ROUGE-L分数
        """
        lcs_length = self.compute_lcs(candidate, reference)
        
        if len(candidate) == 0 or len(reference) == 0:
            return 0.0
        
        # 计算召回率和精确率
        recall = lcs_length / len(reference)
        precision = lcs_length / len(candidate)
        
        # 计算F1分数
        if recall + precision == 0:
            return 0.0
        
        f1 = 2 * recall * precision / (recall + precision)
        return f1
    
    def compute_rouge_l_multiple_refs(self, candidate, references):
        """
        计算多个参考的ROUGE-L分数
        Args:
            candidate: 候选序列
            references: 参考序列列表
        Returns:
            最大ROUGE-L分数
        """
        rouge_scores = []
        for ref in references:
            score = self.compute_rouge_l(candidate, ref)
            rouge_scores.append(score)
        
        return max(rouge_scores)
    
    def compute_batch_rouge_l(self, candidates, references_list):
        """
        批量计算ROUGE-L分数
        Args:
            candidates: 候选序列列表
            references_list: 参考序列列表的列表
        Returns:
            ROUGE-L分数列表
        """
        scores = []
        for candidate, references in zip(candidates, references_list):
            score = self.compute_rouge_l_multiple_refs(candidate, references)
            scores.append(score)
        
        return scores
    
    def compute_rouge_l_ngram(self, candidate, reference, n=1):
        """
        计算n-gram ROUGE-L分数
        Args:
            candidate: 候选序列
            reference: 参考序列
            n: n-gram大小
        Returns:
            n-gram ROUGE-L分数
        """
        # 生成n-gram
        def get_ngrams(seq, n):
            return [tuple(seq[i:i+n]) for i in range(len(seq)-n+1)]
        
        candidate_ngrams = get_ngrams(candidate, n)
        reference_ngrams = get_ngrams(reference, n)
        
        if len(candidate_ngrams) == 0 or len(reference_ngrams) == 0:
            return 0.0
        
        # 计算LCS
        lcs_length = self.compute_lcs(candidate_ngrams, reference_ngrams)
        
        # 计算召回率和精确率
        recall = lcs_length / len(reference_ngrams)
        precision = lcs_length / len(candidate_ngrams)
        
        # 计算F1分数
        if recall + precision == 0:
            return 0.0
        
        f1 = 2 * recall * precision / (recall + precision)
        return f1
