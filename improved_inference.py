#!/usr/bin/env python3
"""
改进的图像描述生成推理脚本
解决半句话问题，提供更好的调试信息
"""

import os
import sys
import torch
from PIL import Image

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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
    print(f"特殊标记索引: SOS={vocab.sos_idx}, EOS={vocab.eos_idx}, PAD={vocab.pad_idx}, UNK={vocab.unk_idx}")
    return model, vocab

def generate_caption_improved(image_path, model, vocab, device, max_length=50, debug=False):
    """改进的描述生成函数，解决半句话问题"""
    # 图像预处理
    transform = ImageTransforms(224, is_training=False)
    
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform.get_transforms()(image).unsqueeze(0).to(device)
        print(f"图像加载成功: {image_path}")
    except Exception as e:
        print(f"图像加载失败: {e}")
        return None
    
    # 生成描述
    with torch.no_grad():
        # 提取图像特征
        features = model.encoder(image_tensor)
        print(f"图像特征形状: {features.shape}")
        
        # 使用解码器的sample方法生成描述
        generated = model.decoder.sample(features, max_length, vocab)
        print(f"生成序列形状: {generated.shape}")
        
        # 转换为文本，添加详细调试信息
        caption = []
        generated_words = []
        
        if debug:
            print("\n=== 生成过程调试信息 ===")
            print("生成的词ID序列:", generated[0].tolist())
        
        for i, word_id in enumerate(generated[0]):
            word = vocab.idx2word[word_id.item()]
            generated_words.append(f"{word}({word_id.item()})")
            
            if debug:
                print(f"步骤 {i+1}: 词ID={word_id.item()}, 词='{word}'")
            
            # 检查是否到达结束标记（使用索引比较更可靠）
            if word_id.item() == vocab.eos_idx:
                if debug:
                    print(f"检测到EOS标记，停止生成")
                break
                
            # 过滤特殊标记
            if word not in [vocab.SOS_TOKEN, vocab.PAD_TOKEN, vocab.UNK_TOKEN]:
                caption.append(word)
            
            # 添加额外的停止条件（如果句子看起来完整）
            if len(caption) > 0 and caption[-1] in ['.', '!', '?']:
                if debug:
                    print(f"检测到句子结束标点，停止生成")
                break
        
        if debug:
            print(f"完整生成序列: {' '.join(generated_words)}")
            print(f"过滤后的描述: {' '.join(caption)}")
        
        result = ' '.join(caption)
        print(f"生成的描述: {result}")
        
        return result

def generate_caption_with_beam_search(image_path, model, vocab, device, beam_size=3, max_length=50):
    """使用束搜索生成描述（可选的高级方法）"""
    # 这里可以实现束搜索，暂时使用贪心搜索
    return generate_caption_improved(image_path, model, vocab, device, max_length, debug=True)

def main():
    """主函数"""
    # 配置
    model_path = "checkpoints/cnn_gru/best_model.pth"
    image_path = "data/DeepFashion-MultiModal/images/MEN-Denim-id_00000080-01_7_additional.jpg"
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 检查文件是否存在
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        print("请先训练模型或检查模型路径")
        return
    
    if not os.path.exists(image_path):
        print(f"错误: 图像文件不存在: {image_path}")
        print("请检查图像路径")
        return
    
    # 加载模型
    model, vocab = load_model(model_path, device)
    
    # 生成描述（带调试信息）
    print(f"\n处理图像: {image_path}")
    print("=" * 50)
    
    caption = generate_caption_improved(
        image_path, model, vocab, device, 
        max_length=50, debug=True
    )
    
    print("=" * 50)
    if caption:
        print(f"最终生成的描述: {caption}")
        
        # 分析生成结果
        words = caption.split()
        print(f"描述长度: {len(words)} 个词")
        print(f"是否以标点结尾: {caption.endswith(('.', '!', '?'))}")
    else:
        print("描述生成失败")

if __name__ == '__main__':
    main()