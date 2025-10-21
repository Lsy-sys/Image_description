#!/usr/bin/env python3
"""
图像描述生成演示脚本
展示如何使用训练好的模型进行图片描述
"""

import os
import sys
import torch
from PIL import Image

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cnn_gru.model import CNNGruModel
from data.transforms import ImageTransforms

class ImageCaptionGenerator:
    """图像描述生成器"""
    
    def __init__(self, model_path, device='auto'):
        """
        初始化描述生成器
        Args:
            model_path: 模型文件路径
            device: 运行设备 ('auto', 'cuda', 'cpu')
        """
        # 设备选择
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"使用设备: {self.device}")
        
        # 加载模型
        self.model, self.vocab = self._load_model(model_path)
        
        # 图像预处理
        self.transform = ImageTransforms(224, is_training=False)
        
        print("模型加载完成！")
    
    def _load_model(self, model_path):
        """加载训练好的模型"""
        print(f"加载模型: {model_path}")
        
        # 加载检查点
        checkpoint = torch.load(model_path, map_location=self.device)
        vocab = checkpoint['vocab']
        
        # 创建模型
        model = CNNGruModel(
            embed_size=512,
            hidden_size=512,
            vocab_size=len(vocab),
            num_layers=1,
            pretrained=True
        ).to(self.device)
        
        # 加载权重
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"词汇表大小: {len(vocab)}")
        return model, vocab
    
    def generate_caption(self, image_path, max_length=20):
        """
        为单张图像生成描述
        Args:
            image_path: 图像文件路径
            max_length: 最大生成长度
        Returns:
            生成的描述文本
        """
        try:
            # 加载和预处理图像
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform.get_transforms()(image).unsqueeze(0).to(self.device)
            
            # 生成描述
            with torch.no_grad():
                # 提取图像特征
                features = self.model.encoder(image_tensor)
                
                # 生成文本序列
                caption = []
                hidden = self.model.decoder.init_hidden(1)
                
                for _ in range(max_length):
                    output, hidden = self.model.decoder(features, hidden)
                    word_id = output.argmax(1)
                    
                    # 检查是否到达结束标记
                    if word_id.item() == self.vocab.word2idx['<eos>']:
                        break
                        
                    # 获取词汇
                    word = self.vocab.idx2word[word_id.item()]
                    
                    # 过滤特殊标记
                    if word not in ['<bos>', '<pad>', '<unk>']:
                        caption.append(word)
                
                return ' '.join(caption)
                
        except Exception as e:
            print(f"处理图像时出错: {e}")
            return None
    
    def batch_generate(self, image_paths, max_length=20):
        """
        批量生成描述
        Args:
            image_paths: 图像文件路径列表
            max_length: 最大生成长度
        Returns:
            描述结果字典
        """
        results = {}
        
        for i, image_path in enumerate(image_paths):
            print(f"处理图像 {i+1}/{len(image_paths)}: {image_path}")
            caption = self.generate_caption(image_path, max_length)
            results[image_path] = caption
            
        return results

def main():
    """主函数 - 演示如何使用"""
    
    # 配置
    model_path = "checkpoints/cnn_gru/best_model.pth"
    image_path = "path/to/your/image.jpg"  # 替换为实际图像路径
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        print("请先训练模型或检查模型路径")
        return
    
    # 创建描述生成器
    generator = ImageCaptionGenerator(model_path)
    
    # 单张图像描述
    if os.path.exists(image_path):
        print(f"\n处理图像: {image_path}")
        caption = generator.generate_caption(image_path)
        
        if caption:
            print(f"生成的描述: {caption}")
        else:
            print("描述生成失败")
    else:
        print(f"图像文件不存在: {image_path}")
        print("请修改 image_path 变量为实际的图像路径")
    
    # 批量处理示例
    # image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]
    # results = generator.batch_generate(image_paths)
    # for path, caption in results.items():
    #     print(f"{path}: {caption}")

if __name__ == '__main__':
    main()
