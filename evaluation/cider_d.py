"""
CIDEr-D评测指标实现
基于TF-IDF的n-gram相似度
"""

import numpy as np
from collections import Counter
import math


class CiderD:
    """CIDEr-D评测指标"""
    
    def __init__(self, n=4, sigma=6.0):
        """
        Args:
            n: n-gram的最大长度
            sigma: 长度惩罚参数
        """
        self.n = n
        self.sigma = sigma
        self.name = "CIDEr-D"
    
    def compute_tf_idf(self, documents, n_gram):
        """
        计算TF-IDF权重
        Args:
            documents: 文档列表
            n_gram: n-gram大小
        Returns:
            TF-IDF权重字典
        """
        # 生成n-gram
        def get_ngrams(doc, n):
            return [tuple(doc[i:i+n]) for i in range(len(doc)-n+1)]
        
        # 统计词频
        term_freq = Counter()
        doc_freq = Counter()
        
        for doc in documents:
            ngrams = get_ngrams(doc, n_gram)
            term_freq.update(ngrams)
            doc_freq.update(set(ngrams))
        
        # 计算TF-IDF
        total_docs = len(documents)
        tf_idf = {}
        
        for term, tf in term_freq.items():
            df = doc_freq[term]
            idf = math.log(total_docs / df)
            tf_idf[term] = tf * idf
        
        return tf_idf
    
    def compute_cider_score(self, candidate, references, n_gram):
        """
        计算CIDEr分数
        Args:
            candidate: 候选序列
            references: 参考序列列表
            n_gram: n-gram大小
        Returns:
            CIDEr分数
        """
        # 生成n-gram
        def get_ngrams(seq, n):
            return [tuple(seq[i:i+n]) for i in range(len(seq)-n+1)]
        
        candidate_ngrams = get_ngrams(candidate, n_gram)
        reference_ngrams = [get_ngrams(ref, n_gram) for ref in references]
        
        if len(candidate_ngrams) == 0:
            return 0.0
        
        # 计算TF-IDF权重
        all_docs = [candidate] + references
        tf_idf = self.compute_tf_idf(all_docs, n_gram)
        
        # 计算候选序列的TF-IDF向量
        candidate_tf_idf = Counter(candidate_ngrams)
        candidate_vector = np.array([tf_idf.get(term, 0) * count 
                                   for term, count in candidate_tf_idf.items()])
        
        # 计算参考序列的TF-IDF向量
        reference_vectors = []
        for ref_ngrams in reference_ngrams:
            ref_tf_idf = Counter(ref_ngrams)
            ref_vector = np.array([tf_idf.get(term, 0) * count 
                                 for term, count in ref_tf_idf.items()])
            reference_vectors.append(ref_vector)
        
        # 计算余弦相似度
        similarities = []
        for ref_vector in reference_vectors:
            if len(candidate_vector) == 0 or len(ref_vector) == 0:
                similarities.append(0.0)
            else:
                # 确保向量长度一致
                max_len = max(len(candidate_vector), len(ref_vector))
                candidate_padded = np.pad(candidate_vector, (0, max_len - len(candidate_vector)))
                ref_padded = np.pad(ref_vector, (0, max_len - len(ref_vector)))
                
                # 计算余弦相似度
                dot_product = np.dot(candidate_padded, ref_padded)
                norm_candidate = np.linalg.norm(candidate_padded)
                norm_ref = np.linalg.norm(ref_padded)
                
                if norm_candidate == 0 or norm_ref == 0:
                    similarities.append(0.0)
                else:
                    similarity = dot_product / (norm_candidate * norm_ref)
                    similarities.append(similarity)
        
        # 返回平均相似度
        return np.mean(similarities)
    
    def compute_length_penalty(self, candidate, references):
        """
        计算长度惩罚
        Args:
            candidate: 候选序列
            references: 参考序列列表
        Returns:
            长度惩罚因子
        """
        candidate_len = len(candidate)
        reference_lens = [len(ref) for ref in references]
        ref_len = np.mean(reference_lens)
        
        if ref_len == 0:
            return 0.0
        
        penalty = math.exp(-((candidate_len - ref_len) ** 2) / (2 * self.sigma ** 2))
        return penalty
    
    def compute_cider_d(self, candidate, references):
        """
        计算CIDEr-D分数
        Args:
            candidate: 候选序列
            references: 参考序列列表
        Returns:
            CIDEr-D分数
        """
        if len(candidate) == 0:
            return 0.0
        
        # 计算1到n-gram的CIDEr分数
        cider_scores = []
        for n in range(1, self.n + 1):
            score = self.compute_cider_score(candidate, references, n)
            cider_scores.append(score)
        
        # 计算平均CIDEr分数
        avg_cider = np.mean(cider_scores)
        
        # 计算长度惩罚
        length_penalty = self.compute_length_penalty(candidate, references)
        
        # 计算最终分数
        cider_d_score = avg_cider * length_penalty
        
        return cider_d_score
    
    def compute_batch_cider_d(self, candidates, references_list):
        """
        批量计算CIDEr-D分数
        Args:
            candidates: 候选序列列表
            references_list: 参考序列列表的列表
        Returns:
            CIDEr-D分数列表
        """
        scores = []
        for candidate, references in zip(candidates, references_list):
            score = self.compute_cider_d(candidate, references)
            scores.append(score)
        
        return scores
