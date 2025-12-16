"""
统一的图像描述生成器接口
连接Encoder和Decoder的forward逻辑
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any


class ImageCaptioner(nn.Module):
    """
    图像描述生成器
    统一的接口，连接视觉编码器和文本解码器
    """
    
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        """
        Args:
            encoder: 视觉编码器
            decoder: 文本解码器
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(
        self,
        images: Optional[torch.Tensor] = None,
        regions: Optional[torch.Tensor] = None,
        captions: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            images: 图像张量 [batch_size, 3, H, W] (用于CNN/ViT编码器)
            regions: 区域特征 [batch_size, num_regions, feature_dim] (用于Region/Graph编码器)
            captions: 目标序列 [batch_size, seq_len] (训练时使用)
            **kwargs: 其他参数
        
        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        # 编码视觉特征
        if images is not None:
            visual_features = self.encoder(images)
        elif regions is not None:
            visual_features = self.encoder(regions)
        else:
            raise ValueError("Either images or regions must be provided")
        
        # 解码生成文本
        if captions is not None:
            # 训练模式：使用teacher forcing
            logits = self.decoder(visual_features, captions, **kwargs)
        else:
            # 推理模式：自回归生成
            logits = self.decoder(visual_features, **kwargs)
        
        return logits
    
    def generate(
        self,
        images: Optional[torch.Tensor] = None,
        regions: Optional[torch.Tensor] = None,
        vocab=None,
        max_length: int = 50,
        strategy: str = 'greedy',
        beam_size: int = 3,
        temperature: float = 1.0,
        top_k: int = 5,
        top_p: float = 0.9,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        生成描述
        
        Args:
            images: 图像张量
            regions: 区域特征
            vocab: 词汇表对象
            max_length: 最大生成长度
            strategy: 解码策略 ('greedy', 'beam', 'sampling')
            beam_size: 束搜索大小
            temperature: 采样温度
            top_k: Top-k采样参数
            top_p: Top-p采样参数
            **kwargs: 其他参数
        
        Returns:
            sequences: [batch_size, seq_len] 生成的序列
            log_probs: [batch_size, seq_len] 对数概率（如果返回）
        """
        # 编码视觉特征
        if images is not None:
            visual_features = self.encoder(images)
        elif regions is not None:
            visual_features = self.encoder(regions)
        else:
            raise ValueError("Either images or regions must be provided")
        
        # 使用解码器生成
        return self.decoder.generate(
            visual_features,
            vocab=vocab,
            max_length=max_length,
            strategy=strategy,
            beam_size=beam_size,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            **kwargs
        )
    
    def encode(self, images: Optional[torch.Tensor] = None, 
               regions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        仅编码视觉特征
        
        Args:
            images: 图像张量
            regions: 区域特征
        
        Returns:
            visual_features: 视觉特征
        """
        if images is not None:
            return self.encoder(images)
        elif regions is not None:
            return self.encoder(regions)
        else:
            raise ValueError("Either images or regions must be provided")


