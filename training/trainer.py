"""
基础训练器
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os
import json
from tqdm import tqdm
import numpy as np

from data.utils import create_data_loader, prepare_teacher_forcing_batch
from data.graph_dataset import graph_collate_fn


class BaseTrainer:
    """基础训练器（新版接口，与 scripts/train.py 对齐）"""
    
    def __init__(
        self,
        model,
        train_dataset,
        val_dataset,
        loss_fn,
        optimizer,
        scheduler,
        device,
        config,
        vocab
    ):
        """
        Args:
            model: 模型（ImageCaptioner）
            train_dataset: 训练数据集
            val_dataset: 验证数据集
            loss_fn: 损失函数（CrossEntropyLoss 或 SCSTLoss 等）
            optimizer: 优化器
            scheduler: 学习率调度器（可为 None）
            device: 设备 (torch.device)
            config: 配置（完整config dict）
            vocab: 词汇表
        """
        self.model = model
        self.vocab = vocab
        self.config = config
        self.device = device
        self.model.to(self.device)
        
        # DataLoader
        batch_size = config['training']['batch_size']
        num_workers = config['data'].get('num_workers', 4)
        # 区分普通图像模型与图模型 (graph_transformer)
        model_type = config['model']['type']
        if model_type == 'graph_transformer':
            # 图模型使用专门的 graph_collate_fn
            self.train_loader = create_data_loader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                collate_fn=graph_collate_fn,
            )
            self.val_loader = create_data_loader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=graph_collate_fn,
            )
        else:
            self.train_loader = create_data_loader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers
            )
            self.val_loader = create_data_loader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers
            )
        
        # 优化器 & 调度器 & 损失
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        
        # 训练状态
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        # 训练日志（用于分析脚本与前端 Training Monitor）
        self.training_log = {
            "epochs": [],
            "train_loss": [],
            "val_loss": [],
        }
        
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        valid_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch+1}')
        
        for batch_idx, batch in enumerate(pbar):
            model_type = self.config['model']['type']

            # 准备数据
            if model_type == 'graph_transformer':
                node_features = batch['node_features'].to(self.device)
                adj_matrix = batch['adj_matrix'].to(self.device)
                images = None
                regions = None
            else:
                if 'images' in batch:
                    images = batch['images'].to(self.device)
                    regions = None
                else:
                    images = None
                    regions = batch['regions'].to(self.device)
            captions = batch['captions']
            
            # 准备输入和目标 (Teacher Forcing)
            input_seqs, target_seqs = prepare_teacher_forcing_batch(batch, self.vocab)
            # 如果没有有效的 captions，跳过该 batch（避免影响平均值）
            if input_seqs.numel() == 0 or target_seqs.numel() == 0:
                pbar.set_postfix({'loss': 'skipped_empty'})
                continue
            input_seqs = input_seqs.to(self.device)
            target_seqs = target_seqs.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            
            if model_type == 'graph_transformer':
                outputs = self.model(
                    node_features=node_features,
                    adj_matrix=adj_matrix,
                    captions=input_seqs,
                )
            else:
                if 'images' in batch:
                    outputs = self.model(images=images, captions=input_seqs)
                else:
                    outputs = self.model(regions=regions, captions=input_seqs)
            
            # 计算损失
            try:
                loss = self.loss_fn(outputs, target_seqs)
            except Exception as e:
                # debug 输出并跳过该 batch
                print("Loss computation error on batch", batch_idx, ":", e)
                print("outputs.shape:", getattr(outputs, 'shape', None))
                print("target_seqs.shape:", target_seqs.shape)
                continue
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            grad_clip = self.config['training'].get('grad_clip', 0.0)
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            
            self.optimizer.step()
            
            # 如果使用按-step 的调度器（如 LambdaLR / CosineAnnealingLR 带 step 语义），在每个 step 后更新
            if self.scheduler and not isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                if isinstance(self.scheduler, (optim.lr_scheduler.LambdaLR, optim.lr_scheduler.CosineAnnealingLR)):
                    try:
                        self.scheduler.step()
                    except Exception:
                        # 如果调度器不支持 per-step，忽略错误（保留兼容性）
                        pass
            # 在训练开始的前若干 step 打印当前学习率以验证 warmup（只在第 0 个 epoch）
            if self.current_epoch == 0 and batch_idx < 10:
                try:
                    current_lr = self.optimizer.param_groups[0].get('lr', None)
                    print(f"[LR CHECK] epoch={self.current_epoch} step={batch_idx} lr={current_lr:.8f}")
                except Exception:
                    pass
            
            total_loss += loss.item()
            valid_batches += 1

            # 更新进度条（展示当前 batch loss）
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            # 首几个 batch 打印 debug 信息，帮助检查 shapes / optimizer 配置
            if batch_idx < 3:
                try:
                    print(f"[DEBUG] batch {batch_idx}: images {None if 'images' not in locals() else images.shape}, "
                          f"input_seqs {input_seqs.shape}, target_seqs {target_seqs.shape}, outputs {getattr(outputs, 'shape', None)}")
                    print(f"[DEBUG] num params total {sum(p.numel() for p in self.model.parameters())}, "
                          f"num params in optimizer {sum(p.numel() for g in self.optimizer.param_groups for p in g['params'])}")
                except Exception:
                    pass
        
        # 平均只对有效 batch 取平均，避免空 batch 干扰
        if valid_batches == 0:
            avg_loss = float('inf')
        else:
            avg_loss = total_loss / valid_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate_epoch(self):
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0
        num_batches = len(self.val_loader)
        valid_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                model_type = self.config['model']['type']

                # 准备数据
                if model_type == 'graph_transformer':
                    node_features = batch['node_features'].to(self.device)
                    adj_matrix = batch['adj_matrix'].to(self.device)
                    images = None
                    regions = None
                else:
                    if 'images' in batch:
                        images = batch['images'].to(self.device)
                        regions = None
                    else:
                        images = None
                        regions = batch['regions'].to(self.device)
                captions = batch['captions']
                
                # 准备输入和目标
                input_seqs, target_seqs = prepare_teacher_forcing_batch(batch, self.vocab)
                input_seqs = input_seqs.to(self.device)
                target_seqs = target_seqs.to(self.device)
                
                # 前向传播
                if model_type == 'graph_transformer':
                    outputs = self.model(
                        node_features=node_features,
                        adj_matrix=adj_matrix,
                        captions=input_seqs,
                    )
                else:
                    if 'images' in batch:
                        outputs = self.model(images=images, captions=input_seqs)
                    else:
                        outputs = self.model(regions=regions, captions=input_seqs)
                
                # 计算损失
                try:
                    loss = self.loss_fn(outputs, target_seqs)
                except Exception as e:
                    print("Validation loss computation error:", e)
                    print("outputs.shape:", getattr(outputs, 'shape', None))
                    print("target_seqs.shape:", target_seqs.shape)
                    continue
                total_loss += loss.item()
                valid_batches += 1
        
        if valid_batches == 0:
            avg_loss = float('inf')
        else:
            avg_loss = total_loss / valid_batches
        self.val_losses.append(avg_loss)
        
        return avg_loss
    
    def train(self):
        """训练模型"""
        total_epochs = self.config['training']['epochs']
        print(f"开始训练，共{total_epochs}个epoch")
        print(f"设备: {self.device}")
        
        for epoch in range(total_epochs):
            self.current_epoch = epoch
            
            # 训练
            train_loss = self.train_epoch()
            
            # 验证
            val_loss = self.validate_epoch()
            
            # 学习率调度：只在 epoch 末对 ReduceLROnPlateau 做基于 epoch 的更新
            if self.scheduler and isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            
            # 记录日志
            self.training_log["epochs"].append(epoch + 1)
            self.training_log["train_loss"].append(float(train_loss))
            self.training_log["val_loss"].append(float(val_loss))

            # 写入日志文件，供 analysis 脚本和前端使用
            log_dir = self.config['paths'].get('log_dir', 'logs')
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, 'training_log.json')
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(self.training_log, f, ensure_ascii=False, indent=2)

            # 打印结果
            print(f'Epoch {epoch+1}/{total_epochs}:')
            print(f'  Train Loss: {train_loss:.4f}')
            print(f'  Val Loss: {val_loss:.4f}')
            print(f'  LR: {self.optimizer.param_groups[0]["lr"]:.6f}')
            
            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(is_best=True)
                print(f'  保存最佳模型 (Val Loss: {val_loss:.4f})')
            
            print('-' * 50)
        
        print("训练完成！")
    
    def save_checkpoint(self, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # 保存最新检查点
        checkpoint_dir = self.config['paths']['checkpoint_dir']
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳检查点
        if is_best:
            best_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"加载检查点: {checkpoint_path}")
        print(f"Epoch: {self.current_epoch}, Best Val Loss: {self.best_val_loss:.4f}")
