#!/usr/bin/env python3
"""
图像描述生成推理脚本
使用训练好的模型对单张图像生成描述
"""

import os
import sys
import argparse
import torch
from PIL import Image

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cnn_gru.model import CNNGruModel
from data.transforms import ImageTransforms

def load_model(model_path, device):
    """加载训练好的模型"""
    print(f"加载模型: {model_path}")
    
    # 加载检查点
    checkpoint = torch.load(model_path, map_location=device)
    vocab = checkpoint['vocab']
    
    # 创建模型
    model = CNNGruModel(
        embed_size=512,
        hidden_size=512,
        vocab_size=len(vocab),
        num_layers=1,
        pretrained=True
    ).to(device)
    
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"模型加载完成，词汇表大小: {len(vocab)}")
    return model, vocab

def generate_caption(image_path, model, vocab, device, max_length=50):
    """为单张图像生成描述"""
    # 图像预处理
    transform = ImageTransforms(224, is_training=False)
    
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform.get_transforms()(image).unsqueeze(0).to(device)
    except Exception as e:
        print(f"图像加载失败: {e}")
        return None
    
    # 生成描述
    with torch.no_grad():
        # 提取图像特征
        features = model.encoder(image_tensor)
        
        # 使用解码器的sample方法生成描述
        generated = model.decoder.sample(features, max_length, vocab)
        
        # 转换为文本
        caption = []
        for word_id in generated[0]:  # 取第一个样本
            word = vocab.idx2word[word_id.item()]
            
            # 检查是否到达结束标记（使用索引比较更可靠）
            if word_id.item() == vocab.eos_idx or word == vocab.EOS_TOKEN:
                break
                
            # 过滤特殊标记
            if word not in [vocab.SOS_TOKEN, vocab.PAD_TOKEN, vocab.UNK_TOKEN]:
                caption.append(word)
        
        return ' '.join(caption)

def main():
    parser = argparse.ArgumentParser(description='图像描述生成推理')
    parser.add_argument('--image', required=True, help='输入图像路径')
    parser.add_argument('--model', default='checkpoints/cnn_gru/best_model.pth', 
                       help='模型文件路径')
    parser.add_argument('--device', default='auto', 
                       choices=['auto', 'cuda', 'cpu'], help='运行设备')
    parser.add_argument('--max_length', type=int, default=50, 
                       help='最大生成长度')
    
    args = parser.parse_args()
    
    # 设备选择
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"使用设备: {device}")
    
    # 检查文件是否存在
    if not os.path.exists(args.image):
        print(f"错误: 图像文件不存在: {args.image}")
        return
    
    if not os.path.exists(args.model):
        print(f"错误: 模型文件不存在: {args.model}")
        print("请先训练模型或检查模型路径")
        return
    
    # 加载模型
    model, vocab = load_model(args.model, device)
    
    # 生成描述
    print(f"\n处理图像: {args.image}")
    caption = generate_caption(args.image, model, vocab, device, args.max_length)
    
    if caption:
        print(f"生成的描述: {caption}")
    else:
        print("描述生成失败")

if __name__ == '__main__':
    main()
