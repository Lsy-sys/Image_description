"""
Transformer编码器
处理区域特征
"""

import torch
import torch.nn as nn
from .attention import MultiHeadAttention, PositionalEncoding


class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层"""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        Args:
            d_model: 模型维度
            num_heads: 注意力头数
            d_ff: 前馈网络维度
            dropout: Dropout概率
        """
        super(TransformerEncoderLayer, self).__init__()
        
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        前向传播
        Args:
            x: 输入 (batch_size, seq_len, d_model)
            mask: 掩码
        Returns:
            输出 (batch_size, seq_len, d_model)
        """
        # 自注意力
        attn_output, _ = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class TransformerEncoder(nn.Module):
    """Transformer编码器"""
    
    def __init__(self, d_model=512, num_heads=8, num_layers=6, d_ff=2048, 
                 dropout=0.1, max_len=100):
        """
        Args:
            d_model: 模型维度
            num_heads: 注意力头数
            num_layers: 编码器层数
            d_ff: 前馈网络维度
            dropout: Dropout概率
            max_len: 最大序列长度
        """
        super(TransformerEncoder, self).__init__()
        
        self.d_model = d_model
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # 编码器层
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        前向传播
        Args:
            x: 输入 (batch_size, seq_len, d_model)
            mask: 掩码
        Returns:
            输出 (batch_size, seq_len, d_model)
        """
        # 位置编码
        x = x * math.sqrt(self.d_model)
        x = self.pos_encoding(x.transpose(0, 1)).transpose(0, 1)
        x = self.dropout(x)
        
        # 通过编码器层
        for layer in self.layers:
            x = layer(x, mask)
        
        return x
