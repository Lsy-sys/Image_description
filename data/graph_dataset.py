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
    ):
        self.data_dir = data_dir
        self.feature_dir = feature_dir
        self.graph_dir = graph_dir
        self.split = split
        self.vocab = vocab

        # 加载样本列表
        list_file = os.path.join(self.data_dir, f"{self.split}_list.txt")
        with open(list_file, "r", encoding="utf-8") as f:
            self.item_ids = [line.strip() for line in f.readlines()]

    def __len__(self) -> int:
        return len(self.item_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item_id = self.item_ids[idx]

        # 节点特征
        node_path = os.path.join(self.feature_dir, f"{item_id}.npy")
        node_features_np = np.load(node_path)  # (num_nodes, node_feature_dim)
        node_features = torch.from_numpy(node_features_np).float()

        # 邻接矩阵（这里使用 spatial 图，如果你构建了 semantic，可以在此扩展）
        adj_path = os.path.join(self.graph_dir, f"{item_id}_spatial.npy")
        adj_np = np.load(adj_path)  # (num_nodes, num_nodes)
        adj_matrix = torch.from_numpy(adj_np).float()

        # 加载文本描述（和 DeepFashionDataset 相同）
        caption_path = os.path.join(self.data_dir, "captions", f"{item_id}.json")
        captions = []
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
                caption_tokens = torch.tensor([self.vocab.pad_idx], dtype=torch.long) if self.vocab else torch.tensor([], dtype=torch.long)
        else:
            caption_tokens = torch.tensor([self.vocab.pad_idx], dtype=torch.long) if self.vocab else torch.tensor([], dtype=torch.long)

        return {
            "node_features": node_features,  # (num_nodes, node_feature_dim)
            "adj_matrix": adj_matrix,        # (num_nodes, num_nodes)
            "caption": caption_tokens,       # (seq_len,)
            "item_id": item_id,
        }


def graph_collate_fn(batch):
    """
    GCN 图数据批处理函数

    - 将节点特征和邻接矩阵按 batch 维堆叠
    - captions 列表用于后续 Teacher Forcing 处理
    """
    node_features = [item["node_features"] for item in batch]
    adj_matrices = [item["adj_matrix"] for item in batch]
    captions = [item["caption"] for item in batch]
    item_ids = [item["item_id"] for item in batch]

    # 假设所有样本 num_nodes 相同（由于统一的预处理），可以直接 stack
    node_features = torch.stack(node_features, dim=0)  # (batch, num_nodes, feat_dim)
    adj_matrices = torch.stack(adj_matrices, dim=0)    # (batch, num_nodes, num_nodes)

    return {
        "node_features": node_features,
        "adj_matrix": adj_matrices,
        "captions": captions,
        "item_ids": item_ids,
    }


