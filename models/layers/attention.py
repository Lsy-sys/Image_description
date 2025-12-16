"""
注意力机制实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        seq_len_q = query.size(1)
        seq_len_k = key.size(1)
        
        Q = self.w_q(query).view(batch_size, seq_len_q, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(1)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            mask = mask.expand(-1, self.num_heads, -1, -1)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, self.d_model
        )
        
        output = self.w_o(context)
        return output, attention_weights


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: (batch_size, seq_len, d_model) 或 (seq_len, batch_size, d_model)
        if x.dim() == 3 and x.size(0) != self.pe.size(0):
            # (batch_size, seq_len, d_model)
            return x + self.pe[:x.size(1), :].unsqueeze(0).transpose(0, 1)
        else:
            # (seq_len, batch_size, d_model)
            return x + self.pe[:x.size(0), :]


class AttentionLayer(nn.Module):
    """简单的注意力层（用于RNN解码器）"""
    
    def __init__(self, embed_size, hidden_size, attention_dim):
        super().__init__()
        self.attention_dim = attention_dim
        
        self.attention_linear = nn.Linear(embed_size, attention_dim)
        self.hidden_linear = nn.Linear(hidden_size, attention_dim)
        self.full_attention = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, features, hidden):
        """
        Args:
            features: 图像特征 (batch_size, num_features, embed_size)
            hidden: 隐藏状态 (batch_size, hidden_size)
        Returns:
            attn_features: 注意力加权的特征 (batch_size, 1, embed_size)
            attn_weights: 注意力权重 (batch_size, num_features)
        """
        # 计算注意力分数
        att1 = self.attention_linear(features)  # (batch_size, num_features, attention_dim)
        att2 = self.hidden_linear(hidden)  # (batch_size, attention_dim)
        att2 = att2.unsqueeze(1)  # (batch_size, 1, attention_dim)
        
        att = self.full_attention(self.relu(att1 + att2)).squeeze(2)  # (batch_size, num_features)
        alpha = self.softmax(att)  # (batch_size, num_features)
        
        # 加权求和
        attn_features = torch.bmm(alpha.unsqueeze(1), features)  # (batch_size, 1, embed_size)
        
        return attn_features, alpha


