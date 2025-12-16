#!/usr/bin/env python3
"""
通用推理脚本（适配当前“配置驱动 + ImageCaptioner”架构）

支持：
- Model A-D：输入单张图片，生成描述
- Model E（Graph-Transformer）：输入 item_id，从 feature_dir / graph_dir 加载 node_features 与 adj_matrix 生成描述
"""

import os
import sys
import argparse
import json
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import create_model_from_config, load_config
from data.vocabulary import Vocabulary
from data.transforms import ImageTransforms


def _resolve_device(device: str) -> torch.device:
    if device == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device)


def _load_vocab(config: dict) -> Vocabulary:
    # 兼容：如果配置里提供 vocab_path，则优先复用；否则从 captions 现建
    vocab_path = config.get('paths', {}).get('vocab_path')
    if vocab_path and os.path.exists(vocab_path):
        return Vocabulary.load(vocab_path)

    data_dir = config['paths']['data_dir']
    min_freq = config.get('data', {}).get('min_freq', 5)
    vocab = Vocabulary.from_captions(data_dir, min_freq=min_freq)

    if vocab_path:
        os.makedirs(os.path.dirname(vocab_path) or '.', exist_ok=True)
        vocab.save(vocab_path)
    return vocab


def _load_checkpoint_into_model(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> None:
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state, strict=True)


def _preprocess_image(image_path: str, image_size: int, device: torch.device) -> torch.Tensor:
    img = Image.open(image_path).convert('RGB')
    transform = ImageTransforms(image_size, is_training=False)
    return transform.get_transforms()(img).unsqueeze(0).to(device)


def _load_graph_inputs(
    item_id: str,
    feature_dir: str,
    graph_dir: str,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    feat_path = os.path.join(feature_dir, f"{item_id}.npy")
    adj_path = os.path.join(graph_dir, f"{item_id}_spatial.npy")

    if not os.path.exists(feat_path):
        raise FileNotFoundError(f"node_features 不存在: {feat_path}")
    if not os.path.exists(adj_path):
        raise FileNotFoundError(f"adj_matrix 不存在: {adj_path}")

    node_features = np.load(feat_path)
    adj_matrix = np.load(adj_path)

    # 先把 numpy 转成 torch
    node_features = torch.from_numpy(node_features).float()
    adj_matrix = torch.from_numpy(adj_matrix).float()

    # -------- 形状修复：尽量兼容 preprocess_features.py 产物的各种形状 --------
    # 目标：node_features -> (1, N, F), adj_matrix -> (1, N, N)
    if adj_matrix.ndim != 2 or adj_matrix.shape[0] != adj_matrix.shape[1]:
        raise ValueError(f"adj_matrix shape 非法: {tuple(adj_matrix.shape)}，期望 (N,N)")
    num_nodes = int(adj_matrix.shape[0])

    # squeeze 掉全 1 维度（例如 (2048,1,1) -> (2048,)）
    if node_features.ndim > 1 and all(d == 1 for d in node_features.shape[1:]):
        node_features = node_features.squeeze()

    # 常见情况：全局特征 (F,) -> 复制成 (N,F)
    if node_features.ndim == 1:
        if node_features.numel() <= 0:
            raise ValueError("node_features 为空")
        node_features = node_features.unsqueeze(0).repeat(num_nodes, 1)

    # 2D：可能是 (N,F) 或 (F,N) 或 (F,1) 等
    if node_features.ndim == 2:
        # (F,1) / (1,F) 这类，压成 (F,)
        if 1 in node_features.shape and max(node_features.shape) == node_features.numel():
            node_features = node_features.reshape(-1).unsqueeze(0).repeat(num_nodes, 1)
        else:
            # 如果是 (F,N) 且 F=2048，更可能是转置的
            if node_features.shape[0] == 2048 and node_features.shape[1] == num_nodes:
                node_features = node_features.t().contiguous()
            # 若是 (N,F) 但 N 不匹配，则尝试截断/填充到 num_nodes
            if node_features.shape[0] != num_nodes:
                if node_features.shape[0] > num_nodes:
                    node_features = node_features[:num_nodes]
                else:
                    pad = torch.zeros((num_nodes - node_features.shape[0], node_features.shape[1]), dtype=node_features.dtype)
                    node_features = torch.cat([node_features, pad], dim=0)

    # 3D：可能已经带 batch，或是奇怪的 (F,1,1) 等；优先 squeeze 再按上面规则处理
    if node_features.ndim == 3:
        # (1,N,F) 直接用
        if node_features.shape[0] == 1 and node_features.shape[1] == num_nodes:
            pass
        else:
            node_features = node_features.squeeze()
            # squeeze 后再递归走一次 1D/2D 逻辑
            if node_features.ndim == 1:
                node_features = node_features.unsqueeze(0).repeat(num_nodes, 1)
            elif node_features.ndim == 2:
                if node_features.shape[0] == 2048 and node_features.shape[1] == num_nodes:
                    node_features = node_features.t().contiguous()
                if node_features.shape[0] != num_nodes:
                    if node_features.shape[0] > num_nodes:
                        node_features = node_features[:num_nodes]
                    else:
                        pad = torch.zeros((num_nodes - node_features.shape[0], node_features.shape[1]), dtype=node_features.dtype)
                        node_features = torch.cat([node_features, pad], dim=0)
            else:
                raise ValueError(f"node_features shape 过于异常: {tuple(node_features.shape)}")

    if node_features.ndim != 2 or node_features.shape[0] != num_nodes:
        raise ValueError(
            f"node_features shape 修复失败: {tuple(node_features.shape)}，期望 (N={num_nodes}, F)"
        )

    # 对齐到 batch 维度
    node_features = node_features.unsqueeze(0)  # (N,F) -> (1,N,F)
    adj_matrix = adj_matrix.unsqueeze(0)        # (N,N) -> (1,N,N)

    return node_features.to(device), adj_matrix.to(device)


def _decode_sequence(seq: torch.Tensor, vocab: Vocabulary) -> str:
    # seq: (T,)
    words = []
    for wid in seq.tolist():
        w = vocab.idx2word.get(int(wid), vocab.UNK_TOKEN)
        if w == vocab.EOS_TOKEN:
            break
        if w not in (vocab.SOS_TOKEN, vocab.PAD_TOKEN, vocab.UNK_TOKEN):
            words.append(w)
    return ' '.join(words)


def main():
    parser = argparse.ArgumentParser(description='通用推理入口（A-E）')
    parser.add_argument('--config', type=str, default='configs/models/1_cnn_gru.yaml',
                        help='模型配置文件路径（如 configs/models/5_graph_trans.yaml）')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='检查点路径（默认使用 config.paths.checkpoint_dir/best_model.pth）')
    parser.add_argument('--image', type=str, default=None, help='输入图像路径（A-D 必填）')
    parser.add_argument('--item_id', type=str, default=None, help='Graph 模型输入 item_id（E 必填）')
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'], help='运行设备')
    parser.add_argument('--max_length', type=int, default=50, help='最大生成长度')
    parser.add_argument('--strategy', type=str, default='greedy', choices=['greedy', 'beam_search', 'sampling'],
                        help='解码策略')
    parser.add_argument('--beam_size', type=int, default=3, help='beam search 大小')
    parser.add_argument('--temperature', type=float, default=1.0, help='采样温度')
    parser.add_argument('--top_k', type=int, default=5, help='top-k 采样')
    parser.add_argument('--top_p', type=float, default=0.9, help='top-p 采样')
    args = parser.parse_args()

    config = load_config(args.config)
    device = _resolve_device(args.device)
    print(f"使用设备: {device}")

    vocab = _load_vocab(config)
    model = create_model_from_config(args.config, len(vocab)).to(device)

    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        ckpt_dir = config.get('paths', {}).get('checkpoint_dir')
        if not ckpt_dir:
            raise ValueError("未提供 --checkpoint 且 config.paths.checkpoint_dir 不存在")
        checkpoint_path = os.path.join(ckpt_dir, 'best_model.pth')

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"检查点不存在: {checkpoint_path}")

    _load_checkpoint_into_model(model, checkpoint_path, device)
    model.eval()

    model_type = config['model']['type']
    image_size = config.get('data', {}).get('image_size', 224)

    with torch.no_grad():
        if model_type == 'graph_transformer':
            if not args.item_id:
                raise ValueError("Graph-Transformer 推理需要 --item_id")
            feature_dir = config['paths'].get('feature_dir', 'data/features')
            graph_dir = config['paths'].get('graph_dir', 'data/graphs')
            node_features, adj_matrix = _load_graph_inputs(args.item_id, feature_dir, graph_dir, device)
            sequences, _ = model.generate(
                node_features=node_features,
                adj_matrix=adj_matrix,
                vocab=vocab,
                max_length=args.max_length,
                strategy=args.strategy,
                beam_size=args.beam_size,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p
            )
        else:
            if not args.image:
                raise ValueError("非 Graph 模型推理需要 --image")
            if not os.path.exists(args.image):
                raise FileNotFoundError(f"图像文件不存在: {args.image}")
            image_tensor = _preprocess_image(args.image, image_size, device)
            sequences, _ = model.generate(
                images=image_tensor,
                vocab=vocab,
                max_length=args.max_length,
                strategy=args.strategy,
                beam_size=args.beam_size,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p
            )

    caption = _decode_sequence(sequences[0].cpu(), vocab)
    print(f"生成的描述: {caption}")


if __name__ == '__main__':
    main()
