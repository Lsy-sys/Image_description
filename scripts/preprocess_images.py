#!/usr/bin/env python3
"""
图像预处理脚本
将原始图像预处理为模型需要的格式
"""

import os
import sys
import argparse
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms as transforms
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def preprocess_images(input_dir, output_dir, image_size=224, quality=95):
    """
    预处理图像文件
    Args:
        input_dir: 输入图像目录
        output_dir: 输出图像目录
        image_size: 目标图像尺寸
        quality: JPEG质量
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 定义图像变换
    transform = transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 获取所有图像文件
    image_files = list(input_path.glob('*.jpg'))
    print(f"找到 {len(image_files)} 个图像文件")
    
    processed_count = 0
    error_count = 0
    
    for image_file in tqdm(image_files, desc="处理图像"):
        try:
            # 打开图像
            with Image.open(image_file) as img:
                # 转换为RGB模式
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # 应用变换
                img_tensor = transform(img)
                
                # 转换回PIL图像用于保存
                # 反归一化
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                img_tensor = img_tensor * std + mean
                img_tensor = torch.clamp(img_tensor, 0, 1)
                
                # 转换为PIL图像
                img_pil = transforms.ToPILImage()(img_tensor)
                
                # 保存处理后的图像
                output_file = output_path / f"{image_file.stem}_processed.jpg"
                img_pil.save(output_file, 'JPEG', quality=quality, optimize=True)
                
                processed_count += 1
                
        except Exception as e:
            print(f"处理图像失败: {image_file} - {e}")
            error_count += 1
    
    print(f"图像预处理完成！")
    print(f"成功处理: {processed_count} 个图像")
    print(f"处理失败: {error_count} 个图像")
    
    return processed_count, error_count

def create_training_transforms():
    """创建训练时的图像变换"""
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

def create_test_transforms():
    """创建测试时的图像变换"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

def main():
    parser = argparse.ArgumentParser(description='预处理图像文件')
    parser.add_argument('--input_dir', type=str, 
                       default='data/DeepFashion-MultiModal/images',
                       help='输入图像目录')
    parser.add_argument('--output_dir', type=str,
                       default='data/DeepFashion-MultiModal/processed_images',
                       help='输出图像目录')
    parser.add_argument('--image_size', type=int, default=224,
                       help='目标图像尺寸')
    parser.add_argument('--quality', type=int, default=95,
                       help='JPEG质量')
    parser.add_argument('--mode', type=str, choices=['preprocess', 'transform'],
                       default='transform', help='处理模式')
    args = parser.parse_args()
    
    print("开始图像预处理...")
    print(f"输入目录: {args.input_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"图像尺寸: {args.image_size}")
    print(f"处理模式: {args.mode}")
    print("=" * 50)
    
    if args.mode == 'preprocess':
        # 预处理模式：调整尺寸和格式
        processed_count, error_count = preprocess_images(
            args.input_dir, 
            args.output_dir, 
            args.image_size, 
            args.quality
        )
    else:
        # 变换模式：只创建变换函数
        print("创建图像变换函数...")
        train_transform = create_training_transforms()
        test_transform = create_test_transforms()
        print("✅ 图像变换函数创建完成！")
        print("\n使用方法:")
        print("from data.transforms import ImageTransforms")
        print("transform = ImageTransforms(image_size=224, is_training=True)")
        print("transformed_img = transform.get_transforms()(image)")
    
    print("\n图像预处理完成！")

if __name__ == '__main__':
    main()
