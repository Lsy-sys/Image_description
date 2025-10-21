#!/usr/bin/env python3
"""
处理图像文件，包括格式转换、尺寸调整等
"""

import os
import sys
import argparse
from PIL import Image
from pathlib import Path
import shutil

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def process_images(input_dir, output_dir, target_size=(224, 224), quality=95):
    """
    处理图像文件
    Args:
        input_dir: 输入图像目录
        output_dir: 输出图像目录
        target_size: 目标尺寸
        quality: JPEG质量
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 支持的图像格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    # 获取所有图像文件
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_path.glob(f'*{ext}'))
        image_files.extend(input_path.glob(f'*{ext.upper()}'))
    
    print(f"找到 {len(image_files)} 个图像文件")
    
    processed_count = 0
    error_count = 0
    
    for image_file in image_files:
        try:
            # 打开图像
            with Image.open(image_file) as img:
                # 转换为RGB模式（处理RGBA等格式）
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # 调整尺寸（保持宽高比）
                img.thumbnail(target_size, Image.Resampling.LANCZOS)
                
                # 创建目标文件路径
                output_file = output_path / f"{image_file.stem}.jpg"
                
                # 保存图像
                img.save(output_file, 'JPEG', quality=quality, optimize=True)
                
                processed_count += 1
                
                if processed_count % 1000 == 0:
                    print(f"已处理 {processed_count} 个图像...")
                    
        except Exception as e:
            print(f"处理图像失败: {image_file} - {e}")
            error_count += 1
    
    print(f"图像处理完成！")
    print(f"成功处理: {processed_count} 个图像")
    print(f"处理失败: {error_count} 个图像")
    
    return processed_count, error_count

def verify_images(image_dir, caption_dir):
    """
    验证图像和描述文件的匹配
    Args:
        image_dir: 图像目录
        caption_dir: 描述目录
    """
    print("验证图像和描述文件匹配...")
    
    image_path = Path(image_dir)
    caption_path = Path(caption_dir)
    
    # 获取图像文件列表
    image_files = set(f.stem for f in image_path.glob('*.jpg'))
    
    # 获取描述文件列表
    caption_files = set(f.stem for f in caption_path.glob('*.json'))
    
    # 检查匹配
    missing_images = caption_files - image_files
    missing_captions = image_files - caption_files
    
    print(f"图像文件数量: {len(image_files)}")
    print(f"描述文件数量: {len(caption_files)}")
    print(f"缺少图像的描述: {len(missing_images)}")
    print(f"缺少描述的图像: {len(missing_captions)}")
    
    if missing_images:
        print("缺少图像的描述文件（前10个）:")
        for img in list(missing_images)[:10]:
            print(f"  - {img}")
    
    if missing_captions:
        print("缺少描述的图像文件（前10个）:")
        for img in list(missing_captions)[:10]:
            print(f"  - {img}")
    
    return len(missing_images) == 0 and len(missing_captions) == 0

def main():
    parser = argparse.ArgumentParser(description='处理图像文件')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='输入图像目录')
    parser.add_argument('--output_dir', type=str, 
                       default='data/DeepFashion-MultiModal/images',
                       help='输出图像目录')
    parser.add_argument('--target_size', type=int, nargs=2, default=[224, 224],
                       help='目标尺寸 (width height)')
    parser.add_argument('--quality', type=int, default=95,
                       help='JPEG质量 (1-100)')
    parser.add_argument('--verify', action='store_true',
                       help='验证图像和描述文件匹配')
    args = parser.parse_args()
    
    print("开始处理图像文件...")
    print(f"输入目录: {args.input_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"目标尺寸: {args.target_size}")
    print(f"JPEG质量: {args.quality}")
    print("=" * 50)
    
    # 处理图像
    processed_count, error_count = process_images(
        args.input_dir, 
        args.output_dir, 
        tuple(args.target_size), 
        args.quality
    )
    
    # 验证匹配
    if args.verify:
        print("\n" + "=" * 50)
        caption_dir = os.path.dirname(args.output_dir) + '/captions'
        is_matched = verify_images(args.output_dir, caption_dir)
        
        if is_matched:
            print("✅ 图像和描述文件完全匹配！")
        else:
            print("⚠️ 图像和描述文件存在不匹配的情况")
    
    print("\n图像处理完成！")

if __name__ == '__main__':
    main()
