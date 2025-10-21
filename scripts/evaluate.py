#!/usr/bin/env python3
"""
模型评测脚本
"""

import os
import sys
import yaml
import argparse
import torch
import json
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import DeepFashionDataset, ImageTransforms, Vocabulary, create_data_loader
from models import CNNGruModel, TransformerModel
from evaluation import compute_metrics


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def load_model(model_type, config, vocab, checkpoint_path):
    """加载模型"""
    if model_type == 'cnn_gru':
        model = CNNGruModel(
            embed_size=config['model']['embed_size'],
            hidden_size=config['model']['hidden_size'],
            vocab_size=vocab.vocab_size,
            num_layers=config['model']['num_layers'],
            dropout=config['model']['dropout'],
            pretrained=config['model']['pretrained']
        )
    elif model_type == 'transformer':
        model = TransformerModel(
            vocab_size=vocab.vocab_size,
            d_model=config['model']['d_model'],
            num_heads=config['model']['num_heads'],
            num_encoder_layers=config['model']['num_encoder_layers'],
            num_decoder_layers=config['model']['num_decoder_layers'],
            d_ff=config['model']['d_ff'],
            dropout=config['model']['dropout'],
            max_len=config['model']['max_len']
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model


def generate_captions(model, data_loader, vocab, device, model_type):
    """生成描述"""
    model.eval()
    model.to(device)
    
    all_candidates = []
    all_references = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Generating captions'):
            if model_type == 'cnn_gru':
                images = batch['images'].to(device)
                generated = model.generate(images, vocab, max_length=20)
            else:  # transformer
                regions = batch['regions'].to(device)
                generated = model.generate(regions, vocab, max_length=20)
            
            # 解码生成的描述
            for i in range(generated.size(0)):
                candidate = vocab.decode(generated[i].tolist())
                all_candidates.append(candidate.split())
            
            # 收集参考描述
            for refs in batch['raw_captions']:
                ref_tokens = [ref.split() for ref in refs]
                all_references.append(ref_tokens)
    
    return all_candidates, all_references


def main():
    parser = argparse.ArgumentParser(description='评测模型')
    parser.add_argument('--model_type', type=str, choices=['cnn_gru', 'transformer'],
                       required=True, help='模型类型')
    parser.add_argument('--config', type=str, required=True,
                       help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='模型检查点路径')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='数据目录')
    parser.add_argument('--vocab_path', type=str, required=True,
                       help='词汇表路径')
    parser.add_argument('--output', type=str, default='evaluation_results.json',
                       help='输出结果文件')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载词汇表
    vocab = Vocabulary()
    vocab.load(args.vocab_path)
    
    # 创建数据变换
    transforms = ImageTransforms(
        image_size=config['data']['image_size'],
        is_training=False
    )
    
    # 创建测试数据集
    if args.model_type == 'cnn_gru':
        test_dataset = DeepFashionDataset(
            data_dir=args.data_dir,
            split='test',
            transform=transforms.get_transforms(),
            vocab=vocab
        )
    else:  # transformer
        from data import RegionDataset
        test_dataset = RegionDataset(
            data_dir=args.data_dir,
            split='test',
            vocab=vocab
        )
    
    # 创建数据加载器
    test_loader = create_data_loader(
        dataset=test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers']
    )
    
    # 加载模型
    model = load_model(args.model_type, config, vocab, args.checkpoint)
    
    # 生成描述
    print("生成描述...")
    candidates, references = generate_captions(
        model, test_loader, vocab, device, args.model_type
    )
    
    # 计算评测指标
    print("计算评测指标...")
    results = compute_metrics(candidates, references, ['rouge_l', 'cider_d'])
    
    # 打印结果
    print("\n评测结果:")
    print(f"ROUGE-L: {results['rouge_l']['mean']:.4f} ± {results['rouge_l']['std']:.4f}")
    print(f"CIDEr-D: {results['cider_d']['mean']:.4f} ± {results['cider_d']['std']:.4f}")
    
    # 保存结果
    output_data = {
        'model_type': args.model_type,
        'checkpoint': args.checkpoint,
        'results': results,
        'num_samples': len(candidates)
    }
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"结果已保存到: {args.output}")


if __name__ == '__main__':
    main()
