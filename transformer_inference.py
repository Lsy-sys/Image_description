#!/usr/bin/env python3
"""
Transformer模型图像描述生成推理脚本
使用区域特征+Transformer架构
"""

import os
import sys
import torch
import torch.nn as nn
from PIL import Image
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.transformer.model import TransformerModel
from data.transforms import ImageTransforms

class RegionFeatureExtractor:
    """区域特征提取器（简化版本）"""
    
    def __init__(self, device):
        self.device = device
        # 这里使用ResNet-50作为特征提取器（简化版本）
        # 在实际应用中，应该使用Faster R-CNN提取区域特征
        import torchvision.models as models
        self.backbone = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])  # 移除最后的分类层
        self.backbone.eval()
        self.backbone.to(device)
        
        # 区域特征维度
        self.feature_dim = 2048
        
    def extract_regions(self, image_tensor, num_regions=36):
        """
        提取图像区域特征
        Args:
            image_tensor: 图像张量 (1, 3, H, W)
            num_regions: 区域数量
        Returns:
            区域特征 (1, num_regions, 2048)
        """
        with torch.no_grad():
            # 提取全局特征
            global_features = self.backbone(image_tensor)  # (1, 2048, 1, 1)
            global_features = global_features.squeeze(-1).squeeze(-1)  # (1, 2048)
            
            # 为了简化，我们复制全局特征作为多个区域特征
            # 在实际应用中，这里应该使用Faster R-CNN提取真正的区域特征
            region_features = global_features.unsqueeze(1).repeat(1, num_regions, 1)  # (1, num_regions, 2048)
            
            # 添加一些随机噪声来模拟不同区域的特征差异
            noise = torch.randn_like(region_features) * 0.1
            region_features = region_features + noise
            
            return region_features

def load_transformer_model(model_path, device):
    """加载训练好的Transformer模型"""
    print(f"加载Transformer模型: {model_path}")
    
    # 加载检查点
    checkpoint = torch.load(model_path, map_location=device)
    vocab = checkpoint['vocab']
    
    # 创建模型
    model = TransformerModel(
        vocab_size=len(vocab),
        d_model=512,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_ff=2048,
        dropout=0.1,
        max_len=100
    ).to(device)
    
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Transformer模型加载完成，词汇表大小: {len(vocab)}")
    print(f"特殊标记索引: SOS={vocab.sos_idx}, EOS={vocab.eos_idx}, PAD={vocab.pad_idx}, UNK={vocab.unk_idx}")
    return model, vocab

def generate_caption_transformer(image_path, model, vocab, region_extractor, device, max_length=50, debug=False):
    """使用Transformer模型生成描述"""
    # 图像预处理
    transform = ImageTransforms(224, is_training=False)
    
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform.get_transforms()(image).unsqueeze(0).to(device)
        print(f"图像加载成功: {image_path}")
    except Exception as e:
        print(f"图像加载失败: {e}")
        return None
    
    # 提取区域特征
    print("提取区域特征...")
    region_features = region_extractor.extract_regions(image_tensor, num_regions=36)
    print(f"区域特征形状: {region_features.shape}")
    
    # 生成描述
    with torch.no_grad():
        # 使用Transformer模型生成描述
        generated = model.generate(region_features, vocab, max_length=max_length, temperature=1.0)
        print(f"生成序列形状: {generated.shape}")
        
        # 转换为文本，添加详细调试信息
        caption = []
        generated_words = []
        
        if debug:
            print("\n=== Transformer生成过程调试信息 ===")
            print("生成的词ID序列:", generated[0].tolist())
        
        for i, word_id in enumerate(generated[0]):
            word = vocab.idx2word[word_id.item()]
            generated_words.append(f"{word}({word_id.item()})")
            
            if debug:
                print(f"步骤 {i+1}: 词ID={word_id.item()}, 词='{word}'")
            
            # 检查是否到达结束标记
            if word_id.item() == vocab.eos_idx:
                if debug:
                    print(f"检测到EOS标记，停止生成")
                break
                
            # 过滤特殊标记
            if word not in [vocab.SOS_TOKEN, vocab.PAD_TOKEN, vocab.UNK_TOKEN]:
                caption.append(word)
            
            # 添加额外的停止条件
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

def main():
    """主函数"""
    # 配置
    model_path = "checkpoints/transformer/best_model.pth"  # 需要训练Transformer模型
    image_path = "data/DeepFashion-MultiModal/images/MEN-Denim-id_00000080-01_7_additional.jpg"
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 检查文件是否存在
    if not os.path.exists(model_path):
        print(f"错误: Transformer模型文件不存在: {model_path}")
        print("请先训练Transformer模型或检查模型路径")
        print("可以使用以下命令训练Transformer模型:")
        print("python scripts/train_transformer.py --config configs/transformer_config.yaml")
        return
    
    if not os.path.exists(image_path):
        print(f"错误: 图像文件不存在: {image_path}")
        print("请检查图像路径")
        return
    
    # 创建区域特征提取器
    region_extractor = RegionFeatureExtractor(device)
    
    # 加载模型
    model, vocab = load_transformer_model(model_path, device)
    
    # 生成描述（带调试信息）
    print(f"\n处理图像: {image_path}")
    print("=" * 60)
    
    caption = generate_caption_transformer(
        image_path, model, vocab, region_extractor, device, 
        max_length=50, debug=True
    )
    
    print("=" * 60)
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
