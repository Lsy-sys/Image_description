"""
基础训练器
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os
from tqdm import tqdm
import numpy as np


class BaseTrainer:
    """基础训练器"""
    
    def __init__(self, model, train_loader, val_loader, vocab, config):
        """
        Args:
            model: 模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            vocab: 词汇表
            config: 配置
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.vocab = vocab
        self.config = config
        
        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # 优化器
        self.optimizer = self._create_optimizer()
        
        # 学习率调度器
        self.scheduler = self._create_scheduler()
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)
        
        # 训练状态
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
    def _create_optimizer(self):
        """创建优化器"""
        if self.config.optimizer == 'adam':
            return optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        elif self.config.optimizer == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)
        elif self.config.optimizer == 'sgd':
            return optim.SGD(self.model.parameters(), lr=self.config.learning_rate, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
    
    def _create_scheduler(self):
        """创建学习率调度器"""
        if self.config.scheduler == 'step':
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        elif self.config.scheduler == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.config.epochs)
        elif self.config.scheduler == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=5)
        else:
            return None
    
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch+1}')
        
        for batch_idx, batch in enumerate(pbar):
            # 准备数据
            if 'images' in batch:
                images = batch['images'].to(self.device)
                captions = batch['captions']
            else:
                regions = batch['regions'].to(self.device)
                captions = batch['captions']
            
            # 准备输入和目标
            input_seqs, target_seqs = self._prepare_batch(captions)
            input_seqs = input_seqs.to(self.device)
            target_seqs = target_seqs.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            
            if 'images' in batch:
                outputs = self.model(images, input_seqs)
            else:
                outputs = self.model(regions, input_seqs)
            
            # 计算损失
            loss = self.criterion(outputs.view(-1, outputs.size(-1)), target_seqs.view(-1))
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            if self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # 更新进度条
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate_epoch(self):
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                # 准备数据
                if 'images' in batch:
                    images = batch['images'].to(self.device)
                    captions = batch['captions']
                else:
                    regions = batch['regions'].to(self.device)
                    captions = batch['captions']
                
                # 准备输入和目标
                input_seqs, target_seqs = self._prepare_batch(captions)
                input_seqs = input_seqs.to(self.device)
                target_seqs = target_seqs.to(self.device)
                
                # 前向传播
                if 'images' in batch:
                    outputs = self.model(images, input_seqs)
                else:
                    outputs = self.model(regions, input_seqs)
                
                # 计算损失
                loss = self.criterion(outputs.view(-1, outputs.size(-1)), target_seqs.view(-1))
                total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        
        return avg_loss
    
    def _prepare_batch(self, captions):
        """准备批次数据（子类需要实现）"""
        raise NotImplementedError
    
    def train(self):
        """训练模型"""
        print(f"开始训练，共{self.config.epochs}个epoch")
        print(f"设备: {self.device}")
        
        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            
            # 训练
            train_loss = self.train_epoch()
            
            # 验证
            val_loss = self.validate_epoch()
            
            # 学习率调度
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # 打印结果
            print(f'Epoch {epoch+1}/{self.config.epochs}:')
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
        checkpoint_path = os.path.join(self.config.checkpoint_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳检查点
        if is_best:
            best_path = os.path.join(self.config.checkpoint_dir, 'best_model.pth')
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
