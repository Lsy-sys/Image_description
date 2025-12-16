"""
实验二：注意力机制可视化
绘制注意力热力图
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2


def visualize_attention(
    image_path: str,
    attention_weights: np.ndarray,
    word: str,
    output_path: str = None
):
    """
    可视化注意力权重
    
    Args:
        image_path: 图像路径
        attention_weights: 注意力权重 (H, W) 或 (num_regions,)
        word: 对应的单词
        output_path: 输出路径
    """
    # 加载图像
    image = Image.open(image_path).convert('RGB')
    img_array = np.array(image)
    
    # 调整注意力权重大小以匹配图像
    if attention_weights.ndim == 1:
        # 区域注意力：需要映射回图像空间
        # 这里简化处理，实际需要根据区域坐标映射
        h, w = img_array.shape[:2]
        attn_map = cv2.resize(
            attention_weights.reshape(-1, 1),
            (w, h),
            interpolation=cv2.INTER_LINEAR
        )
    else:
        attn_map = cv2.resize(
            attention_weights,
            (img_array.shape[1], img_array.shape[0]),
            interpolation=cv2.INTER_LINEAR
        )
    
    # 归一化
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
    
    # 创建热力图
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # 原图
    axes[0].imshow(img_array)
    axes[0].set_title('Original Image', fontsize=14)
    axes[0].axis('off')
    
    # 热力图叠加
    axes[1].imshow(img_array)
    im = axes[1].imshow(attn_map, cmap='jet', alpha=0.5, interpolation='bilinear')
    axes[1].set_title(f'Attention Heatmap for "{word}"', fontsize=14)
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"注意力可视化已保存到: {output_path}")
    else:
        plt.show()


def main():
    """主函数"""
    # 示例：需要从模型推理中获取注意力权重
    print("注意力可视化脚本")
    print("需要从模型推理中提取注意力权重")


if __name__ == '__main__':
    main()



