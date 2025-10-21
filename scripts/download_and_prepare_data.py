#!/usr/bin/env python3
"""
数据下载和预处理脚本
"""

import os
import sys
import json
import requests
import zipfile
import shutil
from pathlib import Path
import argparse
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def download_file(url, filename, chunk_size=8192):
    """下载文件"""
    print(f"正在下载: {filename}")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            size = file.write(chunk)
            progress_bar.update(size)

def extract_zip(zip_path, extract_to):
    """解压ZIP文件"""
    print(f"正在解压: {zip_path}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def create_data_splits(data_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """创建数据分割"""
    print("创建数据分割...")
    
    # 获取所有图像文件
    image_dir = os.path.join(data_dir, 'images')
    if not os.path.exists(image_dir):
        print(f"错误: 图像目录不存在: {image_dir}")
        return
    
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    image_files.sort()
    
    total_images = len(image_files)
    train_size = int(total_images * train_ratio)
    val_size = int(total_images * val_ratio)
    
    # 分割数据
    train_files = image_files[:train_size]
    val_files = image_files[train_size:train_size + val_size]
    test_files = image_files[train_size + val_size:]
    
    # 保存分割列表
    def save_list(files, filename):
        with open(os.path.join(data_dir, filename), 'w', encoding='utf-8') as f:
            for file in files:
                # 去掉文件扩展名作为ID
                file_id = os.path.splitext(file)[0]
                f.write(f"{file_id}\n")
    
    save_list(train_files, 'train_list.txt')
    save_list(val_files, 'val_list.txt')
    save_list(test_files, 'test_list.txt')
    
    print(f"数据分割完成:")
    print(f"  训练集: {len(train_files)} 张图像")
    print(f"  验证集: {len(val_files)} 张图像")
    print(f"  测试集: {len(test_files)} 张图像")

def prepare_caption_files(data_dir):
    """准备描述文件"""
    print("准备描述文件...")
    
    # 创建captions目录
    captions_dir = os.path.join(data_dir, 'captions')
    os.makedirs(captions_dir, exist_ok=True)
    
    # 这里需要根据实际的数据格式来处理
    # 假设原始数据中有描述信息，需要转换为JSON格式
    print("注意: 需要根据实际数据格式手动处理描述文件")
    print("每个描述文件应该包含以下格式的JSON:")
    print('{"captions": ["描述1", "描述2", ...]}')

def extract_region_features(data_dir):
    """提取区域特征（需要预训练的Faster R-CNN）"""
    print("提取区域特征...")
    print("注意: 这需要预训练的Faster R-CNN模型")
    print("建议使用以下代码提取区域特征:")
    print("""
    # 使用预训练的Faster R-CNN提取区域特征
    import torch
    import torchvision
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    
    # 加载预训练模型
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    
    # 对每张图像提取区域特征
    # 详细实现请参考相关代码
    """)

def main():
    parser = argparse.ArgumentParser(description='下载和预处理DeepFashion-MultiModal数据集')
    parser.add_argument('--data_dir', type=str, default='../data/DeepFashion-MultiModal',
                       help='数据目录')
    parser.add_argument('--download', action='store_true',
                       help='是否下载数据')
    parser.add_argument('--prepare', action='store_true',
                       help='是否预处理数据')
    args = parser.parse_args()
    
    # 创建数据目录
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"数据目录: {data_dir.absolute()}")
    
    if args.download:
        print("=" * 50)
        print("数据下载步骤:")
        print("1. 访问官方GitHub: https://github.com/switchablenorms/DeepFashion-MultiModal")
        print("2. 下载数据集文件")
        print("3. 将图像文件放在 images/ 目录下")
        print("4. 将描述文件放在 captions/ 目录下")
        print("=" * 50)
        
        # 这里可以添加自动下载的代码
        # 但由于数据集通常需要申请，建议手动下载
    
    if args.prepare:
        print("=" * 50)
        print("数据预处理...")
        
        # 创建数据分割
        create_data_splits(str(data_dir))
        
        # 准备描述文件
        prepare_caption_files(str(data_dir))
        
        # 提取区域特征
        extract_region_features(str(data_dir))
        
        print("=" * 50)
        print("数据预处理完成！")
        print("下一步:")
        print("1. 检查数据目录结构")
        print("2. 运行训练脚本")
        print("3. 开始模型训练")

if __name__ == '__main__':
    main()
