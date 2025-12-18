"""
RNN解码器：GRU和带注意力的GRU
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from ..layers.attention import AttentionLayer


class GRUDecoder(nn.Module):
    """GRU解码器（用于Model A）"""
    
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, dropout=0.5):
        super().__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        # 词嵌入层
        self.embed = nn.Embedding(vocab_size, embed_size)
        
        # GRU层
        self.gru = nn.GRU(
            embed_size, hidden_size, num_layers,
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 输出层
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, features, captions, lengths=None):
        """
        前向传播（训练时使用teacher forcing）
        Args:
            features: 图像特征 (batch_size, embed_size)
            captions: 输入序列 (batch_size, seq_len)
            lengths: 序列长度（未使用，保持接口一致）
        Returns:
            输出logits (batch_size, seq_len, vocab_size)
        """
        batch_size = features.size(0)
        
        # 词嵌入
        embeddings = self.embed(captions)  # (batch_size, seq_len, embed_size)
        
        # 将图像特征与词嵌入结合
        if embeddings.size(1) > 0:
            image_features = features.unsqueeze(1).expand(-1, embeddings.size(1), -1)
            gru_input = embeddings + image_features
        else:
            gru_input = embeddings
        
        # GRU前向传播
        gru_output, _ = self.gru(gru_input)  # (batch_size, seq_len, hidden_size)
        
        # 输出层
        output = self.linear(self.dropout(gru_output))  # (batch_size, seq_len, vocab_size)
        
        return output
    
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
            visual_features: 视觉特征 (batch_size, embed_size)
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
        
        generated = []
        log_probs_list = []
        input_word = torch.tensor([vocab.word2idx.get('<bos>', 1)] * batch_size, device=device)
        hidden = None
        
        for _ in range(max_length):
            embedded = self.embed(input_word.unsqueeze(1))
            image_features = visual_features.unsqueeze(1)
            gru_input = embedded + image_features
            
            output, hidden = self.gru(gru_input, hidden)
            logits = self.linear(output.squeeze(1)) / temperature
            
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
            input_word = predicted
            
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


class AttnGRUDecoder(nn.Module):
    """带空间注意力的GRU解码器（用于Model B）"""
    
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1,
                 attention_dim=512, dropout=0.5):
        super().__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        # 词嵌入层
        self.embed = nn.Embedding(vocab_size, embed_size)
        
        # 注意力层
        self.attention = AttentionLayer(embed_size, hidden_size, attention_dim)
        
        # GRU层
        self.gru = nn.GRU(
            embed_size + embed_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 输出层
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, features, captions, lengths=None):
        """
        前向传播
        Args:
            features: 图像特征 (batch_size, embed_size) 或 (batch_size, H, W, embed_size)
            captions: 输入序列 (batch_size, seq_len)
        Returns:
            输出logits (batch_size, seq_len, vocab_size)
        """
        batch_size = features.size(0)
        seq_len = captions.size(1)
        
        # 词嵌入
        embeddings = self.embed(captions)  # (batch_size, seq_len, embed_size)
        
        # 初始化隐藏状态
        hidden = None
        outputs = []
        
        for t in range(seq_len):
            # 当前时间步的输入
            current_input = embeddings[:, t:t+1, :]  # (batch_size, 1, embed_size)
            
            # 注意力机制
            if hidden is None:
                # 初始化隐藏状态
                hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size,
                                   device=features.device)
            
            # 计算注意力加权的图像特征
            attn_features, attn_weights = self.attention(
                features.unsqueeze(1) if len(features.shape) == 2 else features,
                hidden[-1]  # 使用最后一层的隐藏状态
            )
            
            # 拼接词嵌入和注意力特征
            gru_input = torch.cat([current_input, attn_features], dim=-1)
            
            # GRU前向传播
            gru_output, hidden = self.gru(gru_input, hidden)
            
            # 输出层
            output = self.linear(self.dropout(gru_output))
            outputs.append(output)
        
        outputs = torch.cat(outputs, dim=1)  # (batch_size, seq_len, vocab_size)
        return outputs
    
    def generate(self, visual_features, vocab=None, max_length=50,
                strategy='greedy', temperature=1.0, **kwargs):
        """生成描述（类似GRUDecoder，但使用注意力）"""
        # 实现类似，但使用注意力机制
        # 为了简化，这里先返回基本实现
        batch_size = visual_features.size(0)
        device = visual_features.device
        
        generated = []
        input_word = torch.tensor([vocab.word2idx.get('<bos>', 1)] * batch_size, device=device)
        hidden = None
        
        for _ in range(max_length):
            embedded = self.embed(input_word.unsqueeze(1))
            
            # 注意力
            if hidden is None:
                hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
            attn_features, _ = self.attention(
                visual_features.unsqueeze(1) if len(visual_features.shape) == 2 else visual_features,
                hidden[-1]
            )
            
            gru_input = torch.cat([embedded, attn_features], dim=-1)
            output, hidden = self.gru(gru_input, hidden)
            logits = self.linear(output.squeeze(1)) / temperature
            
            if strategy == 'sampling':
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                predicted = dist.sample()
            else:
                predicted = logits.argmax(1)
            
            generated.append(predicted)
            input_word = predicted
            
            eos_idx = vocab.word2idx.get('<eos>', 2)
            if (predicted == eos_idx).all():
                break
        
        sequences = torch.stack(generated, dim=1)
        return sequences, None









