"""
图数据集，用于GCN + Transformer模型 (Model E)
从预处理好的节点特征和图结构中读取数据。
"""

import os
from typing import Optional, Dict, Any

import numpy as np
import torch
from torch.utils.data import Dataset


class GraphDataset(Dataset):
    """
    图数据集

    约定：
    - 节点特征存放在: {feature_dir}/{item_id}.npy
      形状: (num_nodes, node_feature_dim)
    - 邻接矩阵存放在: {graph_dir}/{item_id}_spatial.npy
      形状: (num_nodes, num_nodes)
    - 文本描述仍然来自 captions JSON（和 DeepFashionDataset 一致）
    """

    def __init__(
        self,
        data_dir: str,
        feature_dir: str,
        graph_dir: str,
        split: str = "train",
        vocab=None,
        default_num_nodes: int = 196,
        default_node_feature_dim: int = 768,
    ):
        self.data_dir = data_dir
        self.feature_dir = feature_dir
        self.graph_dir = graph_dir
        self.split = split
        self.vocab = vocab
        self.default_num_nodes = default_num_nodes
        self.default_node_feature_dim = default_node_feature_dim

        # 加载样本列表
        list_file = os.path.join(self.data_dir, f"{self.split}_list.txt")
        with open(list_file, "r", encoding="utf-8") as f:
            self.item_ids = [line.strip() for line in f.readlines()]

    def __len__(self) -> int:
        return len(self.item_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item_id = self.item_ids[idx]

        try:
            # 先加载邻接矩阵以获取节点数（如果需要）
            adj_path = os.path.join(self.graph_dir, f"{item_id}_spatial.npy")
            if not os.path.exists(adj_path):
                raise FileNotFoundError(f"Adjacency matrix not found: {adj_path}")
            adj_np = np.load(adj_path)  # (num_nodes, num_nodes)，旧版可能是 36x36，新版是 196x196
            adj_matrix = torch.from_numpy(adj_np).float()

            # --- 统一邻接矩阵形状：全部对齐到 default_num_nodes x default_num_nodes ---
            if adj_matrix.dim() != 2 or adj_matrix.shape[0] != adj_matrix.shape[1]:
                raise ValueError(f"Unexpected adj_matrix shape: {adj_matrix.shape}, expected square (N,N)")

            num_nodes = adj_matrix.shape[0]
            target_nodes = self.default_num_nodes
            if num_nodes != target_nodes:
                # 兼容旧的 36x36 图：截断或零填充到 target_nodes x target_nodes
                new_adj = torch.zeros(target_nodes, target_nodes, dtype=adj_matrix.dtype)
                copy_n = min(num_nodes, target_nodes)
                new_adj[:copy_n, :copy_n] = adj_matrix[:copy_n, :copy_n]
                adj_matrix = new_adj
                num_nodes = target_nodes
            
            # 节点特征
            node_path = os.path.join(self.feature_dir, f"{item_id}.npy")
            if not os.path.exists(node_path):
                raise FileNotFoundError(f"Node features not found: {node_path}")
            node_features_np = np.load(node_path)
            node_features = torch.from_numpy(node_features_np).float()
            
            # 统一处理：确保最终形状为 (num_nodes, node_feature_dim)
            # 处理各种可能的输入形状
            original_shape = node_features.shape
            
            # 如果是1D，先squeeze掉多余的维度
            if node_features.dim() > 2:
                node_features = node_features.squeeze()
            
            # 如果是 (feat_dim, 1) 或 (1, feat_dim)，先处理
            if node_features.dim() == 2:
                if node_features.shape[0] == 1:
                    node_features = node_features.squeeze(0)  # (feat_dim,)
                elif node_features.shape[1] == 1:
                    node_features = node_features.squeeze(1)  # (feat_dim,)
            
            # 现在应该是1D (feat_dim,) 或 2D (num_nodes, feat_dim)
            if node_features.dim() == 1:
                # 一维特征 (feat_dim,)，转换为 (num_nodes, feat_dim)
                feat_dim = node_features.shape[0]
                global_feat = node_features.unsqueeze(0)  # (1, feat_dim)
                node_features = global_feat.repeat(num_nodes, 1)  # (num_nodes, feat_dim)
                # 添加小的随机噪声以区分节点
                noise = torch.randn_like(node_features) * 0.01
                node_features = node_features + noise
            elif node_features.dim() == 2:
                # 二维特征，检查是否是 (num_nodes, feat_dim)
                if node_features.shape[0] != num_nodes:
                    # 如果第一维不是 num_nodes，可能是 (feat_dim, num_nodes) 或其他
                    if node_features.shape[1] == num_nodes:
                        # 转置: (feat_dim, num_nodes) -> (num_nodes, feat_dim)
                        node_features = node_features.t()
                    elif node_features.shape[0] > num_nodes:
                        # 截断
                        node_features = node_features[:num_nodes, :]
                    else:
                        # 填充
                        padding = torch.zeros(num_nodes - node_features.shape[0], node_features.shape[1])
                        node_features = torch.cat([node_features, padding], dim=0)
            else:
                raise ValueError(
                    f"Unexpected node_features shape after processing: {node_features.shape} "
                    f"(original: {original_shape}), expected 1D or 2D"
                )

            # 加载文本描述（和 DeepFashionDataset 相同）
            caption_path = os.path.join(self.data_dir, "captions", f"{item_id}.json")
            if os.path.exists(caption_path):
                import json
                with open(caption_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                captions_raw = data.get("captions", [])
                if captions_raw and self.vocab is not None:
                    # 取一个 caption，用于训练
                    caption_text = captions_raw[0]
                    caption_tokens = torch.tensor(self.vocab.encode(caption_text), dtype=torch.long)
                else:
                    caption_tokens = torch.tensor(
                        [self.vocab.pad_idx], dtype=torch.long
                    ) if self.vocab else torch.tensor([], dtype=torch.long)
            else:
                captions_raw = []
                caption_tokens = torch.tensor(
                    [self.vocab.pad_idx], dtype=torch.long
                ) if self.vocab else torch.tensor([], dtype=torch.long)

            return {
                "node_features": node_features,   # (num_nodes, node_feature_dim)
                "adj_matrix": adj_matrix,         # (num_nodes, num_nodes)
                "caption": caption_tokens,        # (seq_len,)
                "raw_captions": captions_raw,     # List[str]，评测使用
                "item_id": item_id,
            }
        except Exception as e:
            print(f"Error loading item {idx} (item_id={item_id}): {e}")
            # 返回默认值（确保返回字典格式）
            num_nodes = self.default_num_nodes
            node_feature_dim = self.default_node_feature_dim
            default_node_features = torch.zeros(num_nodes, node_feature_dim)
            default_adj_matrix = torch.eye(num_nodes)
            default_caption = torch.tensor(
                [self.vocab.pad_idx], dtype=torch.long
            ) if self.vocab else torch.tensor([], dtype=torch.long)
            
            return {
                "node_features": default_node_features,
                "adj_matrix": default_adj_matrix,
                "caption": default_caption,
                "raw_captions": [],
                "item_id": item_id,
            }


def graph_collate_fn(batch):
    """
    GCN 图数据批处理函数

    - 将节点特征和邻接矩阵按 batch 维堆叠
    - captions 列表用于后续 Teacher Forcing 处理
    """
    node_features_list = [item["node_features"] for item in batch]
    adj_matrices = [item["adj_matrix"] for item in batch]
    captions = [item["caption"] for item in batch]
    raw_captions = [item.get("raw_captions", []) for item in batch]
    item_ids = [item["item_id"] for item in batch]

    # 检查并修复每个样本的 node_features 形状
    # 确保每个都是 (num_nodes, node_feature_dim)
    num_nodes = adj_matrices[0].shape[0]  # 从第一个邻接矩阵获取节点数
    node_feature_dim = None
    
    fixed_node_features = []
    for i, nf in enumerate(node_features_list):
        # 确保是 2D tensor
        if nf.dim() == 1:
            # 一维特征 (feat_dim,)，需要转换为 (num_nodes, feat_dim)
            if node_feature_dim is None:
                node_feature_dim = nf.shape[0]
            nf = nf.unsqueeze(0).repeat(num_nodes, 1)  # (num_nodes, feat_dim)
        elif nf.dim() == 2:
            # 二维特征，检查形状
            if node_feature_dim is None:
                node_feature_dim = nf.shape[1]
            # 确保第一维是 num_nodes
            if nf.shape[0] != num_nodes:
                if nf.shape[0] > num_nodes:
                    nf = nf[:num_nodes, :]
                else:
                    padding = torch.zeros(num_nodes - nf.shape[0], nf.shape[1], dtype=nf.dtype)
                    nf = torch.cat([nf, padding], dim=0)
        else:
            raise ValueError(f"Unexpected node_features shape: {nf.shape}, expected 1D or 2D")
        
        fixed_node_features.append(nf)
    
    # 堆叠成 batch
    node_features = torch.stack(fixed_node_features, dim=0)  # (batch, num_nodes, feat_dim)
    adj_matrices = torch.stack(adj_matrices, dim=0)    # (batch, num_nodes, num_nodes)

    return {
        "node_features": node_features,
        "adj_matrix": adj_matrices,
        "captions": captions,
        "raw_captions": raw_captions,
        "item_ids": item_ids,
    }


