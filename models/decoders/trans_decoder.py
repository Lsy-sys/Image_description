"""
Transformer解码器
用于Model C, D, E
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from ..layers.attention import MultiHeadAttention, PositionalEncoding


class TransformerDecoderLayer(nn.Module):
    """Transformer解码器层"""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
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
        # 自注意力
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
    
    def __init__(self, vocab_size, d_model=512, num_heads=8,
                 num_encoder_layers=6, num_decoder_layers=6,
                 d_ff=2048, dropout=0.1, max_len=100):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # 词嵌入
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # 编码器层（处理视觉特征）
        self.encoder_layers = nn.ModuleList([
            nn.ModuleDict({
                'self_attn': MultiHeadAttention(d_model, num_heads, dropout),
                'ff': nn.Sequential(
                    nn.Linear(d_model, d_ff),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_ff, d_model)
                ),
                'norm1': nn.LayerNorm(d_model),
                'norm2': nn.LayerNorm(d_model),
                'dropout': nn.Dropout(dropout)
            })
            for _ in range(num_encoder_layers)
        ])
        
        # 解码器层
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])
        
        # 输出层
        self.output_projection = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, visual_features, captions=None, tgt_mask=None, src_mask=None):
        """
        前向传播
        Args:
            visual_features: 视觉特征 (batch_size, src_len, d_model)
            captions: 目标序列 (batch_size, tgt_len) - 训练时使用
            tgt_mask: 目标序列掩码
            src_mask: 源序列掩码
        Returns:
            输出logits (batch_size, tgt_len, vocab_size)
        """
        # 编码视觉特征
        x_vis = visual_features
        for layer in self.encoder_layers:
            attn_output, _ = layer['self_attn'](x_vis, x_vis, x_vis, src_mask)
            x_vis = layer['norm1'](x_vis + layer['dropout'](attn_output))
            ff_output = layer['ff'](x_vis)
            x_vis = layer['norm2'](x_vis + layer['dropout'](ff_output))
        
        if captions is not None:
            # 训练模式：使用teacher forcing
            # 词嵌入
            x = self.embedding(captions) * math.sqrt(self.d_model)
            
            # 位置编码
            x = x.transpose(0, 1)  # (seq_len, batch_size, d_model)
            x = self.pos_encoding(x)
            x = x.transpose(0, 1)  # (batch_size, seq_len, d_model)
            x = self.dropout(x)
            
            # 通过解码器层
            for layer in self.decoder_layers:
                x = layer(x, x_vis, tgt_mask, src_mask)
            
            # 输出投影
            output = self.output_projection(x)
            return output
        else:
            # 推理模式：自回归生成
            return x_vis  # 返回编码后的视觉特征，用于生成
    
    def generate(
        self,
        visual_features,
        vocab=None,
        max_length=50,
        strategy='greedy',
        beam_size=3,
        temperature=1.0,
        top_k=5,
        top_p=0.9,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        生成描述
        Args:
            visual_features: 视觉特征 (batch_size, src_len, d_model)
            vocab: 词汇表对象
            max_length: 最大长度
            strategy: 解码策略
            temperature: 采样温度
        Returns:
            sequences: (batch_size, seq_len)
            log_probs: (batch_size, seq_len) 或 None
        """
        batch_size = visual_features.size(0)
        device = visual_features.device
        
        # 编码视觉特征
        x_vis = visual_features
        for layer in self.encoder_layers:
            attn_output, _ = layer['self_attn'](x_vis, x_vis, x_vis, None)
            x_vis = layer['norm1'](x_vis + layer['dropout'](attn_output))
            ff_output = layer['ff'](x_vis)
            x_vis = layer['norm2'](x_vis + layer['dropout'](ff_output))
        
        # 生成序列
        generated = []
        log_probs_list = []
        input_word = torch.tensor([vocab.word2idx.get('<bos>', 1)] * batch_size, device=device)
        
        # 初始化解码器状态
        decoder_input = self.embedding(input_word.unsqueeze(1)) * math.sqrt(self.d_model)
        decoder_input = decoder_input.transpose(0, 1)
        decoder_input = self.pos_encoding(decoder_input)
        decoder_input = decoder_input.transpose(0, 1)
        
        for step in range(max_length):
            # 创建因果掩码
            tgt_mask = self._generate_square_subsequent_mask(decoder_input.size(1)).to(device)
            
            # 通过解码器层
            x = decoder_input
            for layer in self.decoder_layers:
                x = layer(x, x_vis, tgt_mask, None)
            
            # 输出投影
            logits = self.output_projection(x[:, -1, :]) / temperature  # (batch_size, vocab_size)
            
            if strategy == 'sampling':
                probs = F.softmax(logits, dim=-1)
                if top_k > 0:
                    top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)
                    probs = torch.zeros_like(probs)
                    probs.scatter_(1, top_k_indices, top_k_probs)
                    probs = probs / probs.sum(dim=-1, keepdim=True)
                
                dist = torch.distributions.Categorical(probs)
                predicted = dist.sample()
                if log_probs_list is not None:
                    log_probs_list.append(dist.log_prob(predicted))
            else:  # greedy
                predicted = logits.argmax(1)
                if log_probs_list is not None:
                    probs = F.softmax(logits, dim=-1)
                    log_probs = torch.log(probs.gather(1, predicted.unsqueeze(1)).squeeze(1) + 1e-10)
                    log_probs_list.append(log_probs)
            
            generated.append(predicted)
            
            # 更新解码器输入
            next_embed = self.embedding(predicted.unsqueeze(1)) * math.sqrt(self.d_model)
            next_embed = next_embed.transpose(0, 1)
            pos = self.pos_encoding.pe[step+1:step+2, :].unsqueeze(1)
            next_embed = next_embed + pos
            next_embed = next_embed.transpose(0, 1)
            decoder_input = torch.cat([decoder_input, next_embed], dim=1)
            
            # 检查EOS
            eos_idx = vocab.word2idx.get('<eos>', 2)
            if (predicted == eos_idx).all():
                break
        
        sequences = torch.stack(generated, dim=1)
        if log_probs_list:
            while len(log_probs_list) < sequences.size(1):
                log_probs_list.append(torch.zeros(batch_size, device=device))
            log_probs = torch.stack(log_probs_list, dim=1)
            return sequences, log_probs
        return sequences, None
    
    def _generate_square_subsequent_mask(self, sz):
        """生成因果掩码"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask


