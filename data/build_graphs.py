"""
为Model E构建图结构数据
构建语义/空间邻接矩阵
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import argparse

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_spatial_graph(regions, image_size=(224, 224)):
    """
    构建空间邻接矩阵
    基于区域的空间位置关系
    """
    num_regions = len(regions)
    adj_matrix = np.zeros((num_regions, num_regions))
    
    for i in range(num_regions):
        for j in range(i + 1, num_regions):
            # 计算区域间的空间距离
            center_i = np.array([regions[i]['x'] + regions[i]['w']/2,
                               regions[i]['y'] + regions[i]['h']/2])
            center_j = np.array([regions[j]['x'] + regions[j]['w']/2,
                               regions[j]['y'] + regions[j]['h']/2])
            
            distance = np.linalg.norm(center_i - center_j)
            max_distance = np.linalg.norm(image_size)
            
            # 距离越近，权重越大
            weight = np.exp(-distance / (max_distance * 0.3))
            adj_matrix[i, j] = weight
            adj_matrix[j, i] = weight
    
    return adj_matrix


def build_semantic_graph(regions, captions):
    """
    构建语义邻接矩阵
    基于区域的语义相似度
    """
    num_regions = len(regions)
    adj_matrix = np.zeros((num_regions, num_regions))
    
    # 简化实现：基于区域标签的相似度
    for i in range(num_regions):
        for j in range(i + 1, num_regions):
            # 计算语义相似度（简化）
            label_i = regions[i].get('label', '')
            label_j = regions[j].get('label', '')
            
            if label_i == label_j:
                weight = 1.0
            elif label_i and label_j:
                # 计算标签相似度
                weight = 0.5
            else:
                weight = 0.1
            
            adj_matrix[i, j] = weight
            adj_matrix[j, i] = weight
    
    return adj_matrix


def main():
    parser = argparse.ArgumentParser(description='构建图结构数据')
    parser.add_argument('--data_dir', type=str, default='data/DeepFashion-MultiModal',
                       help='数据目录')
    parser.add_argument('--output_dir', type=str, default='data/graphs',
                       help='图数据输出目录')
    parser.add_argument('--graph_type', type=str, default='spatial',
                       choices=['spatial', 'semantic', 'both'],
                       help='图类型')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载数据列表
    train_list_file = os.path.join(args.data_dir, 'train_list.txt')
    with open(train_list_file, 'r') as f:
        image_ids = [line.strip() for line in f.readlines()]
    
    print(f"处理 {len(image_ids)} 个样本")
    
    # 为每个样本构建图
    for image_id in image_ids:
        # 加载区域数据（如果有）
        # 这里简化处理，实际需要从Faster R-CNN结果加载
        
        # 示例：构建随机图结构（实际应该从检测结果构建）
        num_regions = 36
        regions = [
            {
                'x': np.random.randint(0, 200),
                'y': np.random.randint(0, 200),
                'w': np.random.randint(20, 50),
                'h': np.random.randint(20, 50),
                'label': f'region_{i}'
            }
            for i in range(num_regions)
        ]
        
        # 构建邻接矩阵
        if args.graph_type in ['spatial', 'both']:
            spatial_adj = build_spatial_graph(regions)
            np.save(os.path.join(args.output_dir, f"{image_id}_spatial.npy"), spatial_adj)
        
        if args.graph_type in ['semantic', 'both']:
            semantic_adj = build_semantic_graph(regions, [])
            np.save(os.path.join(args.output_dir, f"{image_id}_semantic.npy"), semantic_adj)
    
    print("图结构构建完成！")


if __name__ == '__main__':
    main()


