#!/usr/bin/env python3
"""
改进的图像描述生成推理脚本
解决生成描述不完整的问题
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
    return model, vocab

def generate_caption_improved(image_path, model, vocab, device, max_length=30, temperature=1.0):
    """改进的图像描述生成"""
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
        
        # 手动实现生成过程，增加更多控制
        batch_size = 1
        generated = []
        input_word = torch.tensor([vocab.sos_idx] * batch_size, device=device)
        hidden = None
        
        print(f"开始生成描述 (最大长度: {max_length}):")
        
        for step in range(max_length):
            # 词嵌入
            embedded = model.decoder.embed(input_word.unsqueeze(1))
            
            # 结合图像特征
            image_features = features.unsqueeze(1)
            gru_input = embedded + image_features
            
            # GRU前向传播
            output, hidden = model.decoder.gru(gru_input, hidden)
            
            # 输出层
            logits = model.decoder.linear(output.squeeze(1))
            
            # 应用温度缩放
            if temperature != 1.0:
                logits = logits / temperature
            
            # 获取概率分布
            probs = torch.softmax(logits, dim=1)
            
            # 选择最可能的词
            predicted = probs.argmax(1)
            word = vocab.idx2word[predicted.item()]
            
            print(f"步骤 {step+1}: {word} (概率: {probs[0, predicted.item()]:.4f})")
            
            generated.append(predicted)
            
            # 更新输入
            input_word = predicted
            
            # 如果生成了结束标记，停止
            if predicted.item() == vocab.eos_idx:
                print(f"遇到结束标记，停止生成")
                break
        
        # 转换为文本
        caption = []
        for word_id in generated:
            word = vocab.idx2word[word_id.item()]
            if word == vocab.EOS_TOKEN:
                break
            if word not in [vocab.SOS_TOKEN, vocab.PAD_TOKEN, vocab.UNK_TOKEN]:
                caption.append(word)
        
        return ' '.join(caption)

def generate_multiple_captions(image_path, model, vocab, device, num_captions=3):
    """生成多个候选描述"""
    captions = []
    
    for i in range(num_captions):
        print(f"\n生成候选描述 {i+1}:")
        caption = generate_caption_improved(image_path, model, vocab, device, max_length=25, temperature=0.8)
        if caption:
            captions.append(caption)
            print(f"候选 {i+1}: {caption}")
    
    return captions

def main():
    parser = argparse.ArgumentParser(description='改进的图像描述生成推理')
    parser.add_argument('--image', required=True, help='输入图像路径')
    parser.add_argument('--model', default='checkpoints/cnn_gru/best_model.pth', 
                       help='模型文件路径')
    parser.add_argument('--device', default='auto', 
                       choices=['auto', 'cuda', 'cpu'], help='运行设备')
    parser.add_argument('--max_length', type=int, default=30, 
                       help='最大生成长度')
    parser.add_argument('--temperature', type=float, default=1.0, 
                       help='生成温度')
    parser.add_argument('--multiple', action='store_true', 
                       help='生成多个候选描述')
    
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
        return
    
    # 加载模型
    model, vocab = load_model(args.model, device)
    
    # 生成描述
    print(f"\n处理图像: {args.image}")
    
    if args.multiple:
        captions = generate_multiple_captions(args.image, model, vocab, device)
        print(f"\n所有候选描述:")
        for i, caption in enumerate(captions, 1):
            print(f"{i}. {caption}")
    else:
        caption = generate_caption_improved(args.image, model, vocab, device, 
                                          args.max_length, args.temperature)
        if caption:
            print(f"\n生成的描述: {caption}")
        else:
            print("描述生成失败")

if __name__ == '__main__':
    main()
