#!/usr/bin/env python3
"""
模型评测脚本
支持自动化结果保存路径管理
路径格式: results_evaluation/{model_type}/eval_{split}_{checkpoint_name}.json
"""

import os
import sys
import argparse
import torch
import json
import yaml
from datetime import datetime
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import create_model, load_config
from data import DeepFashionDataset, ImageTransforms, Vocabulary, create_data_loader
from data.graph_dataset import GraphDataset, graph_collate_fn
from evaluation import compute_metrics


def load_model_and_vocab(checkpoint_path, config_path, device):
    """
    加载模型、配置和词汇表
    """
    print(f"加载模型权重: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 1. 获取配置
    if 'config' in checkpoint:
        config = checkpoint['config']
        print("使用 Checkpoint 中的模型配置")
    else:
        if not config_path or not os.path.exists(config_path):
            raise ValueError("Checkpoint 中无配置且未指定有效配置文件路径")
        config = load_config(config_path)
        print(f"加载外部配置: {config_path}")

    # 2. 获取词汇表
    if 'vocab' in checkpoint:
        vocab = checkpoint['vocab']
        print("从 Checkpoint 加载词汇表")
    else:
        vocab_path = config.get('paths', {}).get('vocab_path', 'data/vocab.json')
        if not os.path.exists(vocab_path):
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            possible_path = os.path.join(project_root, vocab_path)
            if os.path.exists(possible_path):
                vocab_path = possible_path

        print(f"尝试加载词汇表文件: {vocab_path}")
        if os.path.exists(vocab_path):
            vocab = Vocabulary.load(vocab_path)
        else:
            raise FileNotFoundError(f"无法找到词汇表文件: {vocab_path}")
            
    # 3. 创建模型
    model = create_model(config, len(vocab))
    model = model.to(device)
    
    # 4. 加载权重
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    return model, vocab, config


def generate_captions(model, data_loader, vocab, device, model_type):
    """生成描述"""
    model.eval()
    all_candidates = []
    all_references = []
    
    print(f"开始生成描述 (Model Type: {model_type})...")
    
    printed_samples = 0
    max_print = 5
    # default behavior: print first encountered samples
    sample_method = getattr(data_loader, 'sample_method', 'sequential')
    sample_set = None
    if sample_method == 'random_unique':
        # data_loader.dataset should exist
        try:
            import random
            n = getattr(data_loader, 'sample_n', 5)
            ds = data_loader.dataset
            total = len(ds)
            indices = random.sample(range(total), min(n, total))
            # collect item ids for chosen indices (if dataset provides item_id)
            chosen_ids = set()
            for idx in indices:
                item = ds[idx]
                iid = item.get('item_id') or item.get('item_ids') or None
                if isinstance(iid, list):
                    iid = iid[0] if iid else None
                if iid is not None:
                    chosen_ids.add(iid)
            sample_set = chosen_ids
            print(f"随机选择要打印的 item_ids (最多 {n}):", sample_set)
        except Exception:
            sample_set = None
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Evaluating'):
            if batch is None:
                continue

            generated = None
            batch_refs = batch['raw_captions']
            
            if model_type == 'graph_transformer':
                node_features = batch['node_features'].to(device)
                adj_matrix = batch['adj_matrix'].to(device)
                generated, _ = model.generate(
                    node_features=node_features,
                    adj_matrix=adj_matrix,
                    vocab=vocab,
                    max_length=50
                )
            else:
                if 'images' in batch and batch['images'] is not None:
                    images = batch['images'].to(device)
                    generated, _ = model.generate(
                        images=images, 
                        vocab=vocab, 
                        max_length=50
                    )
                elif 'regions' in batch and batch['regions'] is not None:
                    regions = batch['regions'].to(device)
                    generated, _ = model.generate(
                        regions=regions, 
                        vocab=vocab, 
                        max_length=50
                    )
            
            if generated is None:
                continue
            
            batch_size = generated.size(0)
            for i in range(batch_size):
                caption_str = vocab.decode(generated[i].tolist())
                all_candidates.append(caption_str.split())
                
                # 打印前若干条样本的生成结果，便于快速检查模型行为
                do_print = False
                if sample_set:
                    # check item id in this batch
                    item_ids = batch.get('item_ids') or batch.get('item_id') or []
                    try:
                        cur_id = item_ids[i]
                    except Exception:
                        cur_id = None
                    if cur_id in sample_set:
                        do_print = True
                        sample_set.discard(cur_id)
                else:
                    if printed_samples < max_print:
                        do_print = True

                if do_print:
                    print("\n" + "-" * 40)
                    print(f"[Sample {printed_samples + 1}] Generated:")
                    print(caption_str)
                    print("References:")
                    for ref in batch_refs[i]:
                        print("  -", ref)
                    printed_samples += 1
            
            for refs in batch_refs:
                ref_tokens = [ref.split() for ref in refs]
                all_references.append(ref_tokens)
    
    return all_candidates, all_references


def save_results(results, args, config, num_samples):
    """
    自动化保存结果逻辑
    路径格式: {result_dir}/{model_type}/eval_{split}_{checkpoint_name}.json
    例如: results_evaluation/region_transformer/eval_test_best_model.json
    """
    model_type = config['model']['type']
    
    # 1. 确定保存目录: results_evaluation/model_type/
    save_dir = os.path.join(args.result_dir, model_type)
    os.makedirs(save_dir, exist_ok=True)
    
    # 2. 确定文件名
    # 提取 checkpoint 文件名 (去除路径和后缀)，例如 "best_model"
    ckpt_name = os.path.basename(args.checkpoint)
    if ckpt_name.endswith('.pth'):
        ckpt_name = ckpt_name[:-4]
    
    # 组合文件名: eval_{split}_{checkpoint_name}.json
    filename = f"eval_{args.split}_{ckpt_name}.json"
    
    save_path = os.path.join(save_dir, filename)
    
    # 3. 准备数据
    output_data = {
        'meta': {
            'model_type': model_type,
            'checkpoint': args.checkpoint,
            'split': args.split,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'num_samples': num_samples
        },
        'results': results,
        'summary': {}
    }
    
    # 添加 summary
    for k, v in results.items():
        if isinstance(v, dict) and 'mean' in v:
            output_data['summary'][k] = round(v['mean'], 4)
        else:
            output_data['summary'][k] = v
    
    # 4. 写入文件
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
        
    return save_path


def main():
    parser = argparse.ArgumentParser(description='评测模型性能')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型权重文件路径 (.pth)')
    parser.add_argument('--config', type=str, default=None, help='配置文件路径')
    
    # 默认保存路径改为 results_evaluation
    parser.add_argument('--result_dir', type=str, default='results_evaluation', help='结果保存根目录')
    
    parser.add_argument('--data_dir', type=str, default='data/DeepFashion-MultiModal', help='数据目录')
    parser.add_argument('--split', type=str, default='test', help='评测数据集 (val 或 test)')
    parser.add_argument('--device', type=str, default='auto', help='运行设备')
    args = parser.parse_args()
    
    # 1. 设置设备
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"使用设备: {device}")
    
    # 2. 加载模型
    try:
        model, vocab, config = load_model_and_vocab(args.checkpoint, args.config, device)
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        return

    model_type = config['model']['type']
    print(f"模型类型: {model_type}")

    # 3. 数据集准备
    if model_type == 'graph_transformer':
        dataset = GraphDataset(
            data_dir=args.data_dir,
            feature_dir=config['paths'].get('feature_dir', 'data/features'),
            graph_dir=config['paths'].get('graph_dir', 'data/graphs'),
            split=args.split,
            vocab=vocab
        )
        collate_fn = graph_collate_fn
    else:
        transforms = ImageTransforms(
            image_size=config['data']['image_size'],
            is_training=False
        )
        dataset = DeepFashionDataset(
            data_dir=args.data_dir,
            split=args.split,
            transform=transforms.get_transforms(),
            vocab=vocab
        )
        from data.utils import collate_fn
    
    # 先随机抽取 5 个不同样本，立即生成并打印（便于快速检查），然后再进行完整评测
    try:
        import random
        n_print = 5
        total = len(dataset)
        indices = random.sample(range(total), min(n_print, total))
        print(f"马上打印 {len(indices)} 个随机样本的生成结果与参考：", indices)
        with torch.no_grad():
            for idx in indices:
                item = dataset[idx]
                item_id = item.get('item_id') or item.get('item_ids') or f"idx_{idx}"
                print("\n" + "=" * 60)
                print(f"Item index: {idx}, item_id: {item_id}")
                refs = item.get('raw_captions', [])
                try:
                    if model_type == 'graph_transformer':
                        node = item['node_features'].unsqueeze(0).to(device)
                        adj = item['adj_matrix'].unsqueeze(0).to(device)
                        seq, _ = model.generate(node_features=node, adj_matrix=adj, vocab=vocab, max_length=50)
                    else:
                        if 'image' in item and item['image'] is not None:
                            img = item['image'].unsqueeze(0).to(device)
                            seq, _ = model.generate(images=img, vocab=vocab, max_length=50)
                        elif 'regions' in item and item['regions'] is not None:
                            reg = item['regions'].unsqueeze(0).to(device)
                            seq, _ = model.generate(regions=reg, vocab=vocab, max_length=50)
                        else:
                            print("无法找到视觉输入，跳过")
                            continue
                    gen = vocab.decode(seq[0].tolist())
                    print("Generated:")
                    print(" ", gen)
                    print("References:")
                    for r in refs:
                        print("  -", r)
                except Exception as e:
                    print("生成失败:", e)
    except Exception as e:
        print("即时打印样本时出错:", e)

    data_loader = create_data_loader(
        dataset=dataset,
        batch_size=config['training'].get('batch_size', 32),
        shuffle=False,
        num_workers=config['data'].get('num_workers', 4),
        collate_fn=collate_fn
    )
    # 固定打印 5 个随机唯一样本用于检查（不作为 CLI 可选项）
    setattr(data_loader, 'sample_method', 'random_unique')
    setattr(data_loader, 'sample_n', 5)
    
    # 4. 生成与计算
    candidates, references = generate_captions(model, data_loader, vocab, device, model_type)
    
    if not candidates:
        print("❌ 警告: 未生成任何描述，请检查数据加载或模型状态")
        return

    print("计算评测指标...")
    metrics_list = ['rouge_l', 'cider_d', 'bleu_1', 'bleu_2', 'bleu_3', 'bleu_4']
    results = compute_metrics(candidates, references, metrics_list)
    
    # 5. 保存结果
    save_path = save_results(results, args, config, len(candidates))
    
    print("\n" + "=" * 40)
    print(f"评测结果 ({model_type} - {args.split}):")
    print("=" * 40)
    for metric, values in results.items():
        if isinstance(values, dict) and 'mean' in values:
            print(f"{metric:<10}: {values['mean']:.4f}")
        else:
            print(f"{metric:<10}: {values:.4f}")
    print("=" * 40)
    print(f"💾 结果已保存至: {save_path}")
    print("=" * 40)


if __name__ == '__main__':
    main()
