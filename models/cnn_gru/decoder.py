"""
GRU解码器
用于生成图像描述文本
"""

import torch
import torch.nn as nn


class GRUDecoder(nn.Module):
    """GRU解码器"""
    
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, dropout=0.5):
        """
        Args:
            embed_size: 嵌入维度
            hidden_size: 隐藏层维度
            vocab_size: 词汇表大小
            num_layers: GRU层数
            dropout: Dropout概率
        """
        super(GRUDecoder, self).__init__()
        
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        # 词嵌入层
        self.embed = nn.Embedding(vocab_size, embed_size)
        
        # GRU层
        self.gru = nn.GRU(embed_size, hidden_size, num_layers, 
                         batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # 输出层
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, features, captions, lengths=None):
        """
        前向传播
        Args:
            features: 图像特征 (batch_size, embed_size)
            captions: 输入序列 (batch_size, seq_len)
            lengths: 序列长度
        Returns:
            输出概率 (batch_size, seq_len, vocab_size)
        """
        batch_size = features.size(0)
        
        # 词嵌入
        embeddings = self.embed(captions)  # (batch_size, seq_len, embed_size)
        
        # 将图像特征作为第一个时间步的输入
        # 在第一个时间步，将图像特征与词嵌入结合
        if embeddings.size(1) > 0:
            # 将图像特征扩展到序列长度
            image_features = features.unsqueeze(1).expand(-1, embeddings.size(1), -1)
            # 结合图像特征和词嵌入
            gru_input = embeddings + image_features
        else:
            gru_input = embeddings
        
        # GRU前向传播
        gru_output, _ = self.gru(gru_input)  # (batch_size, seq_len, hidden_size)
        
        # 输出层
        output = self.linear(self.dropout(gru_output))  # (batch_size, seq_len, vocab_size)
        
        return output
    
    def sample(self, features, max_length=20, vocab=None):
        """
        生成描述（推理时使用）
        Args:
            features: 图像特征 (batch_size, embed_size)
            max_length: 最大生成长度
            vocab: 词汇表对象
        Returns:
            生成的序列
        """
        batch_size = features.size(0)
        device = features.device
        
        # 初始化
        generated = []
        input_word = torch.tensor([vocab.sos_idx] * batch_size, device=device)
        hidden = None
        
        for _ in range(max_length):
            # 词嵌入
            embedded = self.embed(input_word.unsqueeze(1))  # (batch_size, 1, embed_size)
            
            # 结合图像特征
            image_features = features.unsqueeze(1)  # (batch_size, 1, embed_size)
            gru_input = embedded + image_features
            
            # GRU前向传播
            output, hidden = self.gru(gru_input, hidden)
            
            # 输出层
            output = self.linear(output.squeeze(1))  # (batch_size, vocab_size)
            
            # 选择最可能的词
            predicted = output.argmax(1)  # (batch_size,)
            generated.append(predicted)
            
            # 更新输入
            input_word = predicted
            
            # 如果生成了结束标记，停止
            if (predicted == vocab.eos_idx).all():
                break
        
        return torch.stack(generated, dim=1)  # (batch_size, max_length)
