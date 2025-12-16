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
    parser.add_argument('--split', type=str, default='train',
                       choices=['train', 'val', 'test', 'all'],
                       help='生成哪个划分的图：train/val/test/all')
    parser.add_argument('--node_type', type=str, default='regions',
                       choices=['regions', 'vit_patches'],
                       help='节点类型：regions(默认36个区域，随机示例)/vit_patches(14x14=196个patch网格)')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    def _read_list(list_path: str):
        if not os.path.exists(list_path):
            raise FileNotFoundError(f"列表文件不存在: {list_path}")
        with open(list_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f.readlines() if line.strip()]

    # 加载数据列表（支持 train/val/test/all）
    split_to_file = {
        'train': os.path.join(args.data_dir, 'train_list.txt'),
        'val': os.path.join(args.data_dir, 'val_list.txt'),
        'test': os.path.join(args.data_dir, 'test_list.txt'),
    }

    if args.split == 'all':
        image_ids = []
        for split_name, list_path in split_to_file.items():
            ids = _read_list(list_path)
            print(f"{split_name}: {len(ids)}")
            image_ids.extend(ids)
        # 去重保持顺序
        seen = set()
        deduped = []
        for _id in image_ids:
            if _id not in seen:
                seen.add(_id)
                deduped.append(_id)
        image_ids = deduped
    else:
        image_ids = _read_list(split_to_file[args.split])
    
    print(f"处理 {len(image_ids)} 个样本（split={args.split}, graph_type={args.graph_type}）")
    
    def _build_vit_patch_grid_regions(grid_size: int = 14):
        """把 ViT patch 网格当作“regions”，用于复用 build_spatial_graph。"""
        regions = []
        step = 224 / grid_size
        for r in range(grid_size):
            for c in range(grid_size):
                regions.append({
                    'x': float(c * step),
                    'y': float(r * step),
                    'w': float(step),
                    'h': float(step),
                    'label': f'patch_{r}_{c}'
                })
        return regions

    # 为每个样本构建图
    for image_id in image_ids:
        # 加载区域数据（如果有）
        # 这里简化处理，实际需要从Faster R-CNN结果加载
        
        if args.node_type == 'vit_patches':
            # 14x14 patch 网格（196 nodes）
            regions = _build_vit_patch_grid_regions(grid_size=14)
        else:
            # regions：示例随机图结构（实际应该从检测结果构建）
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






