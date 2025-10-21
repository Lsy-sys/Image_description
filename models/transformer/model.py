"""
区域特征+Transformer完整模型
"""

import torch
import torch.nn as nn
from .encoder import TransformerEncoder
from .decoder import TransformerDecoder


class TransformerModel(nn.Module):
    """区域特征+Transformer图像描述生成模型"""
    
    def __init__(self, vocab_size, d_model=512, num_heads=8, num_encoder_layers=6,
                 num_decoder_layers=6, d_ff=2048, dropout=0.1, max_len=100):
        """
        Args:
            vocab_size: 词汇表大小
            d_model: 模型维度
            num_heads: 注意力头数
            num_encoder_layers: 编码器层数
            num_decoder_layers: 解码器层数
            d_ff: 前馈网络维度
            dropout: Dropout概率
            max_len: 最大序列长度
        """
        super(TransformerModel, self).__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # 区域特征投影层
        self.region_projection = nn.Linear(2048, d_model)  # Faster R-CNN特征维度
        
        # 编码器
        self.encoder = TransformerEncoder(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_encoder_layers,
            d_ff=d_ff,
            dropout=dropout,
            max_len=max_len
        )
        
        # 解码器
        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_decoder_layers,
            d_ff=d_ff,
            dropout=dropout,
            max_len=max_len
        )
        
    def forward(self, regions, captions, region_mask=None):
        """
        前向传播
        Args:
            regions: 区域特征 (batch_size, num_regions, 2048)
            captions: 输入序列 (batch_size, seq_len)
            region_mask: 区域掩码
        Returns:
            输出概率 (batch_size, seq_len, vocab_size)
        """
        # 投影区域特征到模型维度
        region_features = self.region_projection(regions)  # (batch_size, num_regions, d_model)
        
        # 编码区域特征
        encoder_output = self.encoder(region_features, region_mask)  # (batch_size, num_regions, d_model)
        
        # 解码生成描述
        outputs = self.decoder(captions, encoder_output)  # (batch_size, seq_len, vocab_size)
        
        return outputs
    
    def generate(self, regions, vocab, max_length=20, temperature=1.0):
        """
        生成图像描述
        Args:
            regions: 区域特征 (batch_size, num_regions, 2048)
            vocab: 词汇表对象
            max_length: 最大生成长度
            temperature: 温度参数
        Returns:
            生成的序列 (batch_size, max_length)
        """
        # 投影区域特征
        region_features = self.region_projection(regions)
        
        # 编码区域特征
        encoder_output = self.encoder(region_features)
        
        # 生成描述
        generated = self.decoder.sample(encoder_output, max_length, vocab, temperature)
        
        return generated
    
    def get_region_features(self, regions):
        """
        获取编码后的区域特征（用于特征分析）
        Args:
            regions: 输入区域特征 (batch_size, num_regions, 2048)
        Returns:
            编码后的区域特征 (batch_size, num_regions, d_model)
        """
        region_features = self.region_projection(regions)
        return self.encoder(region_features)
