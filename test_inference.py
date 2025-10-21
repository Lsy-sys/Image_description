#!/usr/bin/env python3
"""
测试推理脚本
"""

import os
import sys
import torch
from PIL import Image

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.cnn_gru.model import CNNGruModel
from data.transforms import ImageTransforms

def test_inference():
    """测试推理功能"""
    
    # 配置
    model_path = "checkpoints/cnn_gru/best_model.pth"
    image_path = "data/DeepFashion-MultiModal/images/MEN-Denim-id_00000080-01_7_additional.jpg"
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 检查文件是否存在
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        return
    
    if not os.path.exists(image_path):
        print(f"错误: 图像文件不存在: {image_path}")
        return
    
    # 加载模型
    print(f"加载模型: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    vocab = checkpoint['vocab']
    
    model = CNNGruModel(
        embed_size=512,
        hidden_size=512,
        vocab_size=len(vocab),
        num_layers=1,
        pretrained=True
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"模型加载完成，词汇表大小: {len(vocab)}")
    
    # 图像预处理
    transform = ImageTransforms(224, is_training=False)
    
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform.get_transforms()(image).unsqueeze(0).to(device)
        print(f"图像加载成功: {image_path}")
    except Exception as e:
        print(f"图像加载失败: {e}")
        return
    
    # 生成描述
    print("开始生成描述...")
    with torch.no_grad():
        # 提取图像特征
        features = model.encoder(image_tensor)
        print(f"图像特征形状: {features.shape}")
        
        # 使用解码器的sample方法生成描述
        generated = model.decoder.sample(features, max_length=50, vocab=vocab)
        print(f"生成序列形状: {generated.shape}")
        
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
        
        result = ' '.join(caption)
        print(f"生成的描述: {result}")
        
        return result

if __name__ == '__main__':
    test_inference()
