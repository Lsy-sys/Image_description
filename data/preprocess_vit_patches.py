"""
离线提取 ViT patch 级节点特征（用于 Model E: Graph-Transformer）

输出约定：
- {output_dir}/{item_id}.npy
  形状: (196, 768)  # 14x14 patches, hidden_size=768（vit-base）

用法示例：
python data/preprocess_vit_patches.py --data_dir data/DeepFashion-MultiModal --output_dir data/vit_features --split all
"""

import os
import sys
import argparse
from typing import List

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import ViTModel, ViTImageProcessor

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _read_list(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description="提取 ViT patch tokens 作为图节点特征")
    parser.add_argument("--data_dir", type=str, default="data/DeepFashion-MultiModal", help="数据集目录")
    parser.add_argument("--output_dir", type=str, default="data/vit_features", help="输出特征目录")
    parser.add_argument("--model_name", type=str, default="google/vit-base-patch16-224", help="ViT 模型")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test", "all"], help="处理划分")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="运行设备")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    os.makedirs(args.output_dir, exist_ok=True)

    # 读取 item_id 列表
    split_to_file = {
        "train": os.path.join(args.data_dir, "train_list.txt"),
        "val": os.path.join(args.data_dir, "val_list.txt"),
        "test": os.path.join(args.data_dir, "test_list.txt"),
    }
    if args.split == "all":
        item_ids = []
        for s, p in split_to_file.items():
            ids = _read_list(p)
            print(f"{s}: {len(ids)}")
            item_ids.extend(ids)
        # 去重保持顺序
        seen = set()
        deduped = []
        for _id in item_ids:
            if _id not in seen:
                seen.add(_id)
                deduped.append(_id)
        item_ids = deduped
    else:
        item_ids = _read_list(split_to_file[args.split])

    print(f"使用设备: {device}")
    print(f"处理样本数: {len(item_ids)} (split={args.split})")

    processor = ViTImageProcessor.from_pretrained(args.model_name)
    vit = ViTModel.from_pretrained(args.model_name).to(device)
    vit.eval()

    image_dir = os.path.join(args.data_dir, "images")
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"images 目录不存在: {image_dir}")

    with torch.no_grad():
        for item_id in tqdm(item_ids, desc="Extract ViT patch features"):
            out_path = os.path.join(args.output_dir, f"{item_id}.npy")
            if os.path.exists(out_path):
                continue

            # DeepFashion 的图片通常为 .jpg；如果缺失则跳过
            img_path = os.path.join(image_dir, f"{item_id}.jpg")
            if not os.path.exists(img_path):
                # 兼容 .png
                img_path = os.path.join(image_dir, f"{item_id}.png")
                if not os.path.exists(img_path):
                    continue

            try:
                img = Image.open(img_path).convert("RGB")
                inputs = processor(images=img, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(device)

                outputs = vit(pixel_values=pixel_values)
                # (1, 197, 768) = [CLS] + 196 patches
                seq = outputs.last_hidden_state
                patch_tokens = seq[:, 1:, :]  # (1, 196, 768)
                patch_tokens = patch_tokens.squeeze(0).cpu().numpy().astype(np.float32)  # (196, 768)

                np.save(out_path, patch_tokens)
            except Exception as e:
                print(f"处理 {item_id} 出错: {e}")

    print("ViT patch 特征提取完成！")


if __name__ == "__main__":
    main()


