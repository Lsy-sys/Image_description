#!/usr/bin/env python3
"""
数据验证脚本
检查数据集的完整性和格式
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
from PIL import Image

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def check_directory_structure(data_dir):
    """检查目录结构"""
    print("检查目录结构...")
    
    required_dirs = ['images', 'captions', 'regions']
    required_files = ['train_list.txt', 'val_list.txt', 'test_list.txt']
    
    all_good = True
    
    # 检查目录
    for dir_name in required_dirs:
        dir_path = os.path.join(data_dir, dir_name)
        if os.path.exists(dir_path):
            print(f"✓ 目录存在: {dir_name}")
        else:
            print(f"✗ 目录缺失: {dir_name}")
            all_good = False
    
    # 检查文件
    for file_name in required_files:
        file_path = os.path.join(data_dir, file_name)
        if os.path.exists(file_path):
            print(f"✓ 文件存在: {file_name}")
        else:
            print(f"✗ 文件缺失: {file_name}")
            all_good = False
    
    return all_good

def check_images(data_dir, max_check=100):
    """检查图像文件"""
    print(f"\n检查图像文件（最多检查{max_check}张）...")
    
    image_dir = os.path.join(data_dir, 'images')
    if not os.path.exists(image_dir):
        print("✗ 图像目录不存在")
        return False
    
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print("✗ 没有找到图像文件")
        return False
    
    print(f"找到 {len(image_files)} 张图像")
    
    # 检查前几张图像
    checked = 0
    for image_file in image_files[:max_check]:
        image_path = os.path.join(image_dir, image_file)
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                if width < 50 or height < 50:
                    print(f"✗ 图像尺寸过小: {image_file} ({width}x{height})")
                    return False
            checked += 1
        except Exception as e:
            print(f"✗ 图像损坏: {image_file} - {e}")
            return False
    
    print(f"✓ 检查了 {checked} 张图像，全部正常")
    return True

def check_captions(data_dir, max_check=100):
    """检查描述文件"""
    print(f"\n检查描述文件（最多检查{max_check}个）...")
    
    captions_dir = os.path.join(data_dir, 'captions')
    if not os.path.exists(captions_dir):
        print("✗ 描述目录不存在")
        return False
    
    caption_files = [f for f in os.listdir(captions_dir) if f.endswith('.json')]
    
    if not caption_files:
        print("✗ 没有找到描述文件")
        return False
    
    print(f"找到 {len(caption_files)} 个描述文件")
    
    # 检查前几个描述文件
    checked = 0
    for caption_file in caption_files[:max_check]:
        caption_path = os.path.join(captions_dir, caption_file)
        try:
            with open(caption_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if 'captions' not in data:
                print(f"✗ 描述文件格式错误: {caption_file} - 缺少'captions'字段")
                return False
            
            captions = data['captions']
            if not isinstance(captions, list) or len(captions) == 0:
                print(f"✗ 描述文件格式错误: {caption_file} - captions不是非空列表")
                return False
            
            # 检查每个描述
            for i, caption in enumerate(captions):
                if not isinstance(caption, str) or len(caption.strip()) == 0:
                    print(f"✗ 描述文件格式错误: {caption_file} - 第{i+1}个描述无效")
                    return False
            
            checked += 1
        except Exception as e:
            print(f"✗ 描述文件损坏: {caption_file} - {e}")
            return False
    
    print(f"✓ 检查了 {checked} 个描述文件，全部正常")
    return True

def check_regions(data_dir, max_check=100):
    """检查区域特征文件"""
    print(f"\n检查区域特征文件（最多检查{max_check}个）...")
    
    regions_dir = os.path.join(data_dir, 'regions')
    if not os.path.exists(regions_dir):
        print("⚠ 区域特征目录不存在（这是可选的，用于Transformer模型）")
        return True
    
    region_files = [f for f in os.listdir(regions_dir) if f.endswith('.npy')]
    
    if not region_files:
        print("⚠ 没有找到区域特征文件（这是可选的，用于Transformer模型）")
        return True
    
    print(f"找到 {len(region_files)} 个区域特征文件")
    
    # 检查前几个区域特征文件
    checked = 0
    for region_file in region_files[:max_check]:
        region_path = os.path.join(regions_dir, region_file)
        try:
            features = np.load(region_path)
            if len(features.shape) != 2 or features.shape[1] != 2048:
                print(f"✗ 区域特征格式错误: {region_file} - 形状应为(?, 2048)，实际为{features.shape}")
                return False
            checked += 1
        except Exception as e:
            print(f"✗ 区域特征文件损坏: {region_file} - {e}")
            return False
    
    print(f"✓ 检查了 {checked} 个区域特征文件，全部正常")
    return True

def check_data_splits(data_dir):
    """检查数据分割文件"""
    print("\n检查数据分割文件...")
    
    splits = ['train', 'val', 'test']
    all_good = True
    
    for split in splits:
        split_file = os.path.join(data_dir, f'{split}_list.txt')
        if not os.path.exists(split_file):
            print(f"✗ 分割文件不存在: {split}_list.txt")
            all_good = False
            continue
        
        with open(split_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if not lines:
            print(f"✗ 分割文件为空: {split}_list.txt")
            all_good = False
            continue
        
        print(f"✓ {split}集: {len(lines)} 个样本")
    
    return all_good

def main():
    parser = argparse.ArgumentParser(description='验证数据集完整性')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='数据目录路径')
    parser.add_argument('--max_check', type=int, default=100,
                       help='最大检查数量')
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"错误: 数据目录不存在: {data_dir}")
        return
    
    print(f"验证数据集: {data_dir.absolute()}")
    print("=" * 50)
    
    # 执行各项检查
    checks = [
        check_directory_structure(data_dir),
        check_images(data_dir, args.max_check),
        check_captions(data_dir, args.max_check),
        check_regions(data_dir, args.max_check),
        check_data_splits(data_dir)
    ]
    
    print("\n" + "=" * 50)
    print("验证结果:")
    
    if all(checks):
        print("🎉 数据集验证通过！可以开始训练模型。")
    else:
        print("❌ 数据集验证失败，请检查上述问题。")
        print("\n建议:")
        print("1. 检查数据目录结构")
        print("2. 确保图像文件完整")
        print("3. 检查描述文件格式")
        print("4. 运行数据预处理脚本")

if __name__ == '__main__':
    main()
