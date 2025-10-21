"""
CNN+GRU完整模型
"""

import torch
import torch.nn as nn
from .encoder import CNNEncoder
from .decoder import GRUDecoder


class CNNGruModel(nn.Module):
    """CNN+GRU图像描述生成模型"""
    
    def __init__(self, embed_size=512, hidden_size=512, vocab_size=10000, 
                 num_layers=1, dropout=0.5, pretrained=True):
        """
        Args:
            embed_size: 嵌入维度
            hidden_size: 隐藏层维度
            vocab_size: 词汇表大小
            num_layers: GRU层数
            dropout: Dropout概率
            pretrained: 是否使用预训练权重
        """
        super(CNNGruModel, self).__init__()
        
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        # 编码器
        self.encoder = CNNEncoder(embed_size=embed_size, pretrained=pretrained)
        
        # 解码器
        self.decoder = GRUDecoder(
            embed_size=embed_size,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            num_layers=num_layers,
            dropout=dropout
        )
        
    def forward(self, images, captions, lengths=None):
        """
        前向传播
        Args:
            images: 输入图像 (batch_size, 3, H, W)
            captions: 输入序列 (batch_size, seq_len)
            lengths: 序列长度
        Returns:
            输出概率 (batch_size, seq_len, vocab_size)
        """
        # 编码图像
        features = self.encoder(images)  # (batch_size, embed_size)
        
        # 解码生成描述
        outputs = self.decoder(features, captions, lengths)  # (batch_size, seq_len, vocab_size)
        
        return outputs
    
    def generate(self, images, vocab, max_length=20):
        """
        生成图像描述
        Args:
            images: 输入图像 (batch_size, 3, H, W)
            vocab: 词汇表对象
            max_length: 最大生成长度
        Returns:
            生成的序列 (batch_size, max_length)
        """
        # 编码图像
        features = self.encoder(images)  # (batch_size, embed_size)
        
        # 生成描述
        generated = self.decoder.sample(features, max_length, vocab)
        
        return generated
    
    def get_image_features(self, images):
        """
        获取图像特征（用于特征分析）
        Args:
            images: 输入图像 (batch_size, 3, H, W)
        Returns:
            图像特征 (batch_size, embed_size)
        """
        return self.encoder(images)
