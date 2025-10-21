"""
Transformer解码器
生成图像描述文本
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import MultiHeadAttention, PositionalEncoding
import math


class TransformerDecoderLayer(nn.Module):
    """Transformer解码器层"""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        Args:
            d_model: 模型维度
            num_heads: 注意力头数
            d_ff: 前馈网络维度
            dropout: Dropout概率
        """
        super(TransformerDecoderLayer, self).__init__()
        
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, encoder_output, tgt_mask=None, src_mask=None):
        """
        前向传播
        Args:
            x: 输入 (batch_size, tgt_len, d_model)
            encoder_output: 编码器输出 (batch_size, src_len, d_model)
            tgt_mask: 目标序列掩码
            src_mask: 源序列掩码
        Returns:
            输出 (batch_size, tgt_len, d_model)
        """
        # 自注意力（带掩码）
        attn_output, _ = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 交叉注意力
        cross_attn_output, _ = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        # 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x


class TransformerDecoder(nn.Module):
    """Transformer解码器"""
    
    def __init__(self, vocab_size, d_model=512, num_heads=8, num_layers=6, 
                 d_ff=2048, dropout=0.1, max_len=100):
        """
        Args:
            vocab_size: 词汇表大小
            d_model: 模型维度
            num_heads: 注意力头数
            num_layers: 解码器层数
            d_ff: 前馈网络维度
            dropout: Dropout概率
            max_len: 最大序列长度
        """
        super(TransformerDecoder, self).__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # 词嵌入
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # 解码器层
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # 输出层
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, tgt, encoder_output, tgt_mask=None, src_mask=None):
        """
        前向传播
        Args:
            tgt: 目标序列 (batch_size, tgt_len)
            encoder_output: 编码器输出 (batch_size, src_len, d_model)
            tgt_mask: 目标序列掩码
            src_mask: 源序列掩码
        Returns:
            输出 (batch_size, tgt_len, vocab_size)
        """
        # 词嵌入
        x = self.embedding(tgt) * math.sqrt(self.d_model)
        
        # 位置编码
        x = self.pos_encoding(x.transpose(0, 1)).transpose(0, 1)
        x = self.dropout(x)
        
        # 通过解码器层
        for layer in self.layers:
            x = layer(x, encoder_output, tgt_mask, src_mask)
        
        # 输出投影
        output = self.output_projection(x)
        
        return output
    
    def generate_square_subsequent_mask(self, sz):
        """生成因果掩码"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def sample(self, encoder_output, max_length=20, vocab=None, temperature=1.0):
        """
        生成描述（推理时使用）
        Args:
            encoder_output: 编码器输出 (batch_size, src_len, d_model)
            max_length: 最大生成长度
            vocab: 词汇表对象
            temperature: 温度参数
        Returns:
            生成的序列 (batch_size, max_length)
        """
        batch_size = encoder_output.size(0)
        device = encoder_output.device
        
        # 初始化
        generated = []
        input_word = torch.tensor([vocab.sos_idx] * batch_size, device=device)
        
        for _ in range(max_length):
            # 准备输入序列
            if generated:
                input_seq = torch.cat([torch.tensor([[vocab.sos_idx]] * batch_size, device=device), 
                                     torch.stack(generated, dim=1)], dim=1)
            else:
                input_seq = torch.tensor([[vocab.sos_idx]] * batch_size, device=device)
            
            # 生成掩码
            tgt_mask = self.generate_square_subsequent_mask(input_seq.size(1)).to(device)
            
            # 前向传播
            output = self.forward(input_seq, encoder_output, tgt_mask)
            
            # 获取最后一个时间步的输出
            next_token_logits = output[:, -1, :] / temperature
            
            # 选择下一个词
            if temperature == 0:
                next_token = next_token_logits.argmax(dim=-1)
            else:
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1).squeeze(1)
            
            generated.append(next_token)
            
            # 如果生成了结束标记，停止
            if (next_token == vocab.eos_idx).all():
                break
        
        return torch.stack(generated, dim=1)
