"""
离线提取图像特征
为Region和Graph模型预处理特征文件
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def extract_resnet_features(image_path, model, transform, device):
    """提取ResNet特征"""
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        features = model(image_tensor)
        features = features.squeeze(0).cpu().numpy()
    
    return features


def extract_faster_rcnn_features(image_path, model, transform, device):
    """提取Faster R-CNN区域特征"""
    # 这里需要实际的Faster R-CNN模型
    # 简化实现：使用ResNet特征作为替代
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # 实际应该使用Faster R-CNN提取36个区域特征
        # 这里简化处理
        features = model(image_tensor)
        if isinstance(features, tuple):
            features = features[0]
        features = features.squeeze(0).cpu().numpy()
    
    return features


def main():
    parser = argparse.ArgumentParser(description='预处理图像特征')
    parser.add_argument('--data_dir', type=str, default='data/DeepFashion-MultiModal',
                       help='数据目录')
    parser.add_argument('--output_dir', type=str, default='data/features',
                       help='特征输出目录')
    parser.add_argument('--model_type', type=str, default='resnet',
                       choices=['resnet', 'faster_rcnn'],
                       help='特征提取模型类型')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--device', type=str, default='auto',
                       help='运行设备')
    
    args = parser.parse_args()
    
    # 确定设备
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"使用设备: {device}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载模型
    if args.model_type == 'resnet':
        model = models.resnet50(pretrained=True)
        model = nn.Sequential(*list(model.children())[:-1])  # 移除最后的FC层
        model.eval().to(device)
    else:
        # Faster R-CNN需要单独实现
        print("Faster R-CNN特征提取需要单独实现")
        return
    
    # 图像变换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 加载图像列表
    image_dir = os.path.join(args.data_dir, 'images')
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
    
    print(f"找到 {len(image_files)} 张图像")
    
    # 提取特征
    for image_file in tqdm(image_files, desc='提取特征'):
        image_path = os.path.join(image_dir, image_file)
        feature_file = os.path.join(args.output_dir, f"{os.path.splitext(image_file)[0]}.npy")
        
        if os.path.exists(feature_file):
            continue
        
        try:
            if args.model_type == 'resnet':
                features = extract_resnet_features(image_path, model, transform, device)
            else:
                features = extract_faster_rcnn_features(image_path, model, transform, device)
            
            np.save(feature_file, features)
        except Exception as e:
            print(f"处理 {image_file} 时出错: {e}")
    
    print("特征提取完成！")


if __name__ == '__main__':
    main()











