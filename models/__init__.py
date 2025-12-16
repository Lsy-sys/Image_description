"""
模型工厂：根据配置动态创建模型实例
"""

import yaml
import os
from typing import Dict, Any, Optional
import torch.nn as nn

from .captioner import ImageCaptioner


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """递归合并字典，override 优先"""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_update(result[k], v)
        else:
            result[k] = v
    return result


def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件，支持继承，使用深度合并"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    if '_base_' in config:
        base_path = config['_base_']
        if not os.path.isabs(base_path):
            base_path = os.path.join(os.path.dirname(config_path), base_path)
        base_config = load_config(base_path)
        merged = _deep_update(base_config, {k: v for k, v in config.items() if k != '_base_'})
        return merged
    
    return config


def create_model(config: Dict[str, Any], vocab_size: int) -> nn.Module:
    """
    根据配置创建模型
    
    Args:
        config: 模型配置字典
        vocab_size: 词汇表大小
    
    Returns:
        模型实例
    """
    model_type = config['model']['type']
    
    if model_type == 'cnn_gru':
        from .encoders.cnn_encoder import ResNetEncoder
        from .decoders.rnn_decoder import GRUDecoder
        from .captioner import ImageCaptioner
        
        encoder = ResNetEncoder(
            embed_size=config['model']['encoder']['embed_size'],
            pretrained=config['model']['encoder']['pretrained']
        )
        decoder = GRUDecoder(
            embed_size=config['model']['encoder']['embed_size'],
            hidden_size=config['model']['decoder']['hidden_size'],
            vocab_size=vocab_size,
            num_layers=config['model']['decoder']['num_layers'],
            dropout=config['model']['decoder'].get('dropout', 0.5)
        )
        return ImageCaptioner(encoder, decoder)
    
    elif model_type == 'attn_gru':
        from .encoders.cnn_encoder import ResNetEncoder
        from .decoders.rnn_decoder import AttnGRUDecoder
        from .captioner import ImageCaptioner
        
        encoder = ResNetEncoder(
            embed_size=config['model']['encoder']['embed_size'],
            pretrained=config['model']['encoder']['pretrained']
        )
        decoder = AttnGRUDecoder(
            embed_size=config['model']['encoder']['embed_size'],
            hidden_size=config['model']['decoder']['hidden_size'],
            vocab_size=vocab_size,
            num_layers=config['model']['decoder']['num_layers'],
            attention_dim=config['model']['decoder']['attention_dim'],
            dropout=config['model']['decoder'].get('dropout', 0.5)
        )
        return ImageCaptioner(encoder, decoder)
    
    elif model_type == 'region_transformer':
        from .encoders.region_encoder import FasterRCNNEncoder
        from .decoders.trans_decoder import TransformerDecoder
        from .captioner import ImageCaptioner
        
        encoder = FasterRCNNEncoder(
            feature_dim=config['model']['encoder']['feature_dim'],
            max_regions=config['model']['encoder']['max_regions']
        )
        decoder = TransformerDecoder(
            vocab_size=vocab_size,
            d_model=config['model']['decoder']['d_model'],
            num_heads=config['model']['decoder']['num_heads'],
            num_encoder_layers=config['model']['decoder']['num_encoder_layers'],
            num_decoder_layers=config['model']['decoder']['num_decoder_layers'],
            d_ff=config['model']['decoder']['d_ff'],
            dropout=config['model']['decoder']['dropout'],
            max_len=config['model']['decoder']['max_len']
        )
        return ImageCaptioner(encoder, decoder)
    
    elif model_type == 'vit_transformer':
        from .encoders.vit_encoder import ViTEncoder
        from .decoders.trans_decoder import TransformerDecoder
        from .captioner import ImageCaptioner
        
        encoder = ViTEncoder(
            model_name=config['model']['encoder']['model_name'],
            feature_dim=config['model']['encoder']['feature_dim'],
            patch_size=config['model']['encoder']['patch_size']
        )
        decoder = TransformerDecoder(
            vocab_size=vocab_size,
            d_model=config['model']['decoder']['d_model'],
            num_heads=config['model']['decoder']['num_heads'],
            num_encoder_layers=config['model']['decoder']['num_encoder_layers'],
            num_decoder_layers=config['model']['decoder']['num_decoder_layers'],
            d_ff=config['model']['decoder']['d_ff'],
            dropout=config['model']['decoder']['dropout'],
            max_len=config['model']['decoder']['max_len']
        )
        return ImageCaptioner(encoder, decoder)
    
    elif model_type == 'graph_transformer':
        from .encoders.graph_encoder import GCNEncoder
        from .decoders.trans_decoder import TransformerDecoder
        from .captioner import ImageCaptioner
        
        encoder = GCNEncoder(
            node_feature_dim=config['model']['encoder']['node_feature_dim'],
            edge_feature_dim=config['model']['encoder']['edge_feature_dim'],
            hidden_dim=config['model']['encoder']['hidden_dim'],
            num_layers=config['model']['encoder']['num_layers'],
            dropout=config['model']['encoder']['dropout']
        )
        decoder = TransformerDecoder(
            vocab_size=vocab_size,
            d_model=config['model']['decoder']['d_model'],
            num_heads=config['model']['decoder']['num_heads'],
            num_encoder_layers=config['model']['decoder']['num_encoder_layers'],
            num_decoder_layers=config['model']['decoder']['num_decoder_layers'],
            d_ff=config['model']['decoder']['d_ff'],
            dropout=config['model']['decoder']['dropout'],
            max_len=config['model']['decoder']['max_len']
        )
        return ImageCaptioner(encoder, decoder)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def create_model_from_config(config_path: str, vocab_size: int) -> nn.Module:
    """
    从配置文件创建模型
    
    Args:
        config_path: 配置文件路径
        vocab_size: 词汇表大小
    
    Returns:
        模型实例
    """
    config = load_config(config_path)
    return create_model(config, vocab_size)
