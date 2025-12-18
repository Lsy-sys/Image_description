"""
词嵌入层
"""

import torch
import torch.nn as nn


class WordEmbedding(nn.Module):
    """词嵌入层（带位置编码）"""
    
    def __init__(self, vocab_size, embed_dim, padding_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.embed_dim = embed_dim
    
    def forward(self, x):
        """
        Args:
            x: 输入序列 (batch_size, seq_len)
        Returns:
            嵌入向量 (batch_size, seq_len, embed_dim)
        """
        return self.embedding(x) * (self.embed_dim ** 0.5)









