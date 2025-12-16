"""
图卷积网络 (GCN) 编码器
用于Model E，处理节点和边信息
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class GCNLayer(nn.Module):
    """图卷积层"""
    
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_dim)
    
    def forward(self, node_features: torch.Tensor, adj_matrix: torch.Tensor):
        """
        Args:
            node_features: 节点特征 (batch_size, num_nodes, in_dim)
            adj_matrix: 邻接矩阵 (batch_size, num_nodes, num_nodes)
        Returns:
            更新后的节点特征 (batch_size, num_nodes, out_dim)
        """
        # 图卷积：A * X * W
        # 归一化邻接矩阵
        adj_norm = F.softmax(adj_matrix, dim=-1)
        
        # 消息传递
        aggregated = torch.bmm(adj_norm, node_features)  # (batch_size, num_nodes, in_dim)
        
        # 线性变换
        output = self.linear(aggregated)  # (batch_size, num_nodes, out_dim)
        output = self.norm(output)
        output = self.dropout(output)
        
        return output


class GCNEncoder(nn.Module):
    """图卷积网络编码器"""
    
    def __init__(
        self,
        node_feature_dim: int = 2048,
        edge_feature_dim: int = 128,
        hidden_dim: int = 512,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        """
        Args:
            node_feature_dim: 节点特征维度
            edge_feature_dim: 边特征维度
            hidden_dim: 隐藏层维度
            num_layers: GCN层数
            dropout: Dropout概率
        """
        super().__init__()
        
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 节点特征投影
        self.node_proj = nn.Linear(node_feature_dim, hidden_dim)
        
        # GCN层
        self.gcn_layers = nn.ModuleList([
            GCNLayer(hidden_dim, hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # 边特征处理（可选）
        if edge_feature_dim > 0:
            self.edge_proj = nn.Linear(edge_feature_dim, hidden_dim)
    
    def forward(
        self,
        node_features: torch.Tensor,
        adj_matrix: torch.Tensor,
        edge_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        Args:
            node_features: 节点特征 (batch_size, num_nodes, node_feature_dim)
            adj_matrix: 邻接矩阵 (batch_size, num_nodes, num_nodes)
            edge_features: 边特征 (batch_size, num_nodes, num_nodes, edge_feature_dim) (可选)
        Returns:
            编码后的节点特征 (batch_size, num_nodes, hidden_dim)
        """
        # 检查并修复 node_features 的形状
        batch_size = adj_matrix.shape[0]
        num_nodes = adj_matrix.shape[1]
        
        if node_features.dim() == 3:
            # 期望的形状: (batch_size, num_nodes, node_feature_dim)
            if node_features.shape[0] != batch_size or node_features.shape[1] != num_nodes:
                # 如果形状不匹配，尝试修复
                if node_features.shape[0] == batch_size * num_nodes:
                    # 可能是被错误地 flatten 了: (batch*num_nodes, feat_dim)
                    node_features = node_features.view(batch_size, num_nodes, -1)
                else:
                    raise ValueError(
                        f"node_features shape {node_features.shape} does not match "
                        f"expected (batch_size={batch_size}, num_nodes={num_nodes}, ...)"
                    )
        elif node_features.dim() == 2:
            # 可能是 (batch*num_nodes, feat_dim)，需要 reshape
            if node_features.shape[0] == batch_size * num_nodes:
                node_features = node_features.view(batch_size, num_nodes, -1)
            else:
                raise ValueError(
                    f"node_features shape {node_features.shape} cannot be reshaped to "
                    f"(batch_size={batch_size}, num_nodes={num_nodes}, ...)"
                )
        else:
            raise ValueError(
                f"node_features must be 2D or 3D, got {node_features.dim()}D with shape {node_features.shape}"
            )
        
        # 确保特征维度匹配
        if node_features.shape[2] != self.node_feature_dim:
            # 如果特征维度不匹配，可能需要投影或报错
            if node_features.shape[2] == 1:
                # 特殊情况：如果最后一维是1，可能是被错误地处理了
                raise ValueError(
                    f"node_features has wrong feature dimension: {node_features.shape[2]}, "
                    f"expected {self.node_feature_dim}. Shape: {node_features.shape}"
                )
            # 如果维度不匹配但不是1，可能需要添加投影层
            raise ValueError(
                f"node_features feature dimension {node_features.shape[2]} does not match "
                f"expected {self.node_feature_dim}"
            )
        
        # 投影节点特征
        x = self.node_proj(node_features)  # (batch_size, num_nodes, hidden_dim)
        
        # 如果提供了边特征，可以融合到邻接矩阵中
        if edge_features is not None:
            # 简化处理：将边特征投影并加到邻接矩阵
            edge_proj = self.edge_proj(edge_features)  # (batch_size, num_nodes, num_nodes, hidden_dim)
            # 这里简化处理，实际可以更复杂
            adj_matrix = adj_matrix + edge_proj.mean(dim=-1)
        
        # 多层GCN
        for gcn_layer in self.gcn_layers:
            x = gcn_layer(x, adj_matrix)
            x = F.relu(x)
        
        return x






