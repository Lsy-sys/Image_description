"""
基于强化学习的损失函数
实现Self-Critical Sequence Training (SCST)算法
直接优化评测指标（BLEU、CIDEr-D、ROUGE-L等）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple

from evaluation import compute_metrics


class RLLoss(nn.Module):
    """
    强化学习损失函数
    基于REINFORCE算法和Self-Critical Sequence Training (SCST)
    """
    
    def __init__(self, reward_type='cider_d', baseline_type='self_critical', 
                 temperature=1.0, sample_size=1):
        """
        Args:
            reward_type: 奖励类型 ('cider_d', 'rouge_l', 'bleu_4', 'combined')
            baseline_type: 基线类型 ('self_critical', 'average', 'none')
            temperature: 采样温度
            sample_size: 每个样本的采样数量（用于平均奖励）
        """
        super(RLLoss, self).__init__()
        self.reward_type = reward_type
        self.baseline_type = baseline_type
        self.temperature = temperature
        self.sample_size = sample_size
    
    def compute_reward(self, candidates: List[List[str]], 
                      references_list: List[List[List[str]]]) -> torch.Tensor:
        """
        计算奖励
        Args:
            candidates: 候选序列列表，每个是词列表
            references_list: 参考序列列表的列表，每个参考也是词列表
        Returns:
            奖励张量 (batch_size,)
        """
        batch_size = len(candidates)
        rewards = torch.zeros(batch_size, dtype=torch.float32)
        
        if self.reward_type == 'cider_d':
            # 使用CIDEr-D作为奖励
            from evaluation import CiderD
            cider = CiderD()
            scores = cider.compute_batch_cider_d(candidates, references_list)
            rewards = torch.tensor(scores, dtype=torch.float32)
        
        elif self.reward_type == 'rouge_l':
            # 使用ROUGE-L作为奖励
            from evaluation import RougeL
            rouge = RougeL()
            scores = rouge.compute_batch_rouge_l(candidates, references_list)
            rewards = torch.tensor(scores, dtype=torch.float32)
        
        elif self.reward_type == 'bleu_4':
            # 使用BLEU-4作为奖励
            from evaluation import BLEU
            bleu = BLEU(max_n=4)
            scores = bleu.compute_batch_bleu(candidates, references_list)
            rewards = torch.tensor(scores, dtype=torch.float32)
        
        elif self.reward_type == 'combined':
            # 组合多种奖励
            from evaluation import CiderD, RougeL, BLEU
            
            cider = CiderD()
            rouge = RougeL()
            bleu = BLEU(max_n=4)
            
            cider_scores = cider.compute_batch_cider_d(candidates, references_list)
            rouge_scores = rouge.compute_batch_rouge_l(candidates, references_list)
            bleu_scores = bleu.compute_batch_bleu(candidates, references_list)
            
            # 归一化并组合（CIDEr-D权重更高，因为它范围更大）
            cider_tensor = torch.tensor(cider_scores, dtype=torch.float32)
            rouge_tensor = torch.tensor(rouge_scores, dtype=torch.float32)
            bleu_tensor = torch.tensor(bleu_scores, dtype=torch.float32)
            
            # 归一化到[0, 1]
            cider_norm = cider_tensor / (cider_tensor.max() + 1e-8)
            rouge_norm = rouge_tensor / (rouge_tensor.max() + 1e-8)
            bleu_norm = bleu_tensor / (bleu_tensor.max() + 1e-8)
            
            # 组合权重：CIDEr-D: 0.5, ROUGE-L: 0.3, BLEU-4: 0.2
            rewards = 0.5 * cider_norm + 0.3 * rouge_norm + 0.2 * bleu_norm
        
        else:
            raise ValueError(f"Unknown reward type: {self.reward_type}")
        
        return rewards
    
    def sample_sequences(self, model, images_or_regions, vocab, 
                        max_length: int, model_type='cnn_gru') -> Tuple[torch.Tensor, torch.Tensor]:
        """
        使用模型采样序列并计算对数概率
        Args:
            model: 模型对象
            images_or_regions: 图像或区域特征
            vocab: 词汇表
            max_length: 最大长度
            model_type: 模型类型 ('cnn_gru' 或 'transformer')
        Returns:
            sampled_seqs: 采样的序列 (batch_size, max_length)
            log_probs: 对数概率 (batch_size, max_length)
        """
        # 使用模型的sample方法进行采样
        if model_type == 'cnn_gru':
            sampled_seqs, log_probs = model.decoder.sample(
                images_or_regions, 
                max_length=max_length, 
                vocab=vocab,
                temperature=self.temperature,
                return_probs=True,
                sample=True
            )
        elif model_type == 'transformer':
            # Transformer模型的采样
            sampled_seqs, log_probs = model.generate(
                images_or_regions,
                vocab,
                max_length=max_length,
                temperature=self.temperature,
                return_probs=True,
                sample=True
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return sampled_seqs, log_probs
    
    def get_baseline_sequences(self, model, images_or_regions, vocab,
                              max_length: int, model_type='cnn_gru') -> torch.Tensor:
        """
        获取基线序列（greedy解码）
        Args:
            model: 模型对象
            images_or_regions: 图像或区域特征
            vocab: 词汇表
            max_length: 最大长度
            model_type: 模型类型
        Returns:
            baseline_seqs: 基线序列 (batch_size, max_length)
        """
        if model_type == 'cnn_gru':
            baseline_seqs = model.decoder.sample(
                images_or_regions,
                max_length=max_length,
                vocab=vocab,
                temperature=1.0,
                return_probs=False,
                sample=False  # greedy
            )
        elif model_type == 'transformer':
            baseline_seqs = model.generate(
                images_or_regions,
                vocab,
                max_length=max_length,
                temperature=1.0,
                return_probs=False,
                sample=False
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return baseline_seqs
    
    def forward(self, sampled_seqs: torch.Tensor, log_probs: torch.Tensor,
                references_list: List[List[List[str]]], vocab,
                baseline_seqs: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        计算强化学习损失
        Args:
            sampled_seqs: 采样的序列 (batch_size, seq_len)
            log_probs: 采样序列的对数概率 (batch_size, seq_len)
            references_list: 参考序列列表
            vocab: 词汇表
            baseline_seqs: 基线序列（用于self-critical，通常是greedy解码）
        Returns:
            损失字典，包含 'loss', 'reward', 'baseline', 'advantage' 等
        """
        batch_size = sampled_seqs.size(0)
        device = sampled_seqs.device
        
        # 将序列转换为词列表
        candidates = []
        for i in range(batch_size):
            seq = sampled_seqs[i].cpu().tolist()
            words = vocab.decode(seq)
            candidates.append(words.split() if words else [])
        
        # 计算采样序列的奖励
        sampled_rewards = self.compute_reward(candidates, references_list).to(device)
        
        # 计算基线
        if self.baseline_type == 'self_critical':
            if baseline_seqs is None:
                # 如果没有提供基线序列，使用平均奖励作为基线
                baseline = sampled_rewards.mean()
            else:
                # 将基线序列转换为词列表
                baseline_candidates = []
                for i in range(batch_size):
                    seq = baseline_seqs[i].cpu().tolist()
                    words = vocab.decode(seq)
                    baseline_candidates.append(words.split() if words else [])
                
                # 计算基线奖励
                baseline_rewards = self.compute_reward(baseline_candidates, references_list).to(device)
                baseline = baseline_rewards
        elif self.baseline_type == 'average':
            baseline = sampled_rewards.mean()
        else:  # 'none'
            baseline = torch.zeros_like(sampled_rewards)
        
        # 计算优势（advantage）
        if self.baseline_type == 'self_critical' and baseline_seqs is not None:
            advantages = sampled_rewards - baseline
        else:
            advantages = sampled_rewards - baseline
        
        # 计算每个时间步的总对数概率
        # 只考虑非padding的位置
        seq_mask = (sampled_seqs != vocab.pad_idx) & (sampled_seqs != vocab.eos_idx)
        seq_log_probs = (log_probs * seq_mask.float()).sum(dim=1)  # (batch_size,)
        
        # REINFORCE损失：-log_prob * advantage
        # 我们最大化奖励，所以损失是负的
        loss = -(seq_log_probs * advantages).mean()
        
        # 额外的熵正则化（鼓励探索）
        entropy_reg = -torch.mean(log_probs * torch.exp(log_probs))
        
        return {
            'loss': loss,
            'reward': sampled_rewards.mean().item(),
            'baseline': baseline.mean().item() if isinstance(baseline, torch.Tensor) else baseline.item(),
            'advantage': advantages.mean().item(),
            'entropy': entropy_reg.item(),
            'perplexity': torch.exp(seq_log_probs.mean()).item()
        }


class MixedLoss(nn.Module):
    """
    混合损失函数：结合交叉熵损失和强化学习损失
    用于微调阶段，可以逐步从交叉熵损失过渡到RL损失
    """
    
    def __init__(self, rl_loss: RLLoss, ce_weight: float = 0.5, rl_weight: float = 0.5):
        """
        Args:
            rl_loss: 强化学习损失函数
            ce_weight: 交叉熵损失权重
            rl_weight: 强化学习损失权重
        """
        super(MixedLoss, self).__init__()
        self.rl_loss = rl_loss
        self.ce_weight = ce_weight
        self.rl_weight = rl_weight
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=0)  # 假设0是padding索引
    
    def forward(self, outputs: torch.Tensor, targets: torch.Tensor,
                sampled_seqs: torch.Tensor, log_probs: torch.Tensor,
                references_list: List[List[List[str]]], vocab,
                baseline_seqs: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        计算混合损失
        Args:
            outputs: 模型输出logits (batch_size, seq_len, vocab_size)
            targets: 目标序列 (batch_size, seq_len)
            sampled_seqs: 采样的序列 (batch_size, seq_len)
            log_probs: 采样序列的对数概率 (batch_size, seq_len)
            references_list: 参考序列列表
            vocab: 词汇表
            baseline_seqs: 基线序列
        Returns:
            损失字典
        """
        # 交叉熵损失
        ce_loss_value = self.ce_loss(
            outputs.view(-1, outputs.size(-1)),
            targets.view(-1)
        )
        
        # 强化学习损失
        rl_results = self.rl_loss(
            sampled_seqs, log_probs, references_list, vocab, baseline_seqs
        )
        
        # 组合损失
        total_loss = self.ce_weight * ce_loss_value + self.rl_weight * rl_results['loss']
        
        return {
            'loss': total_loss,
            'ce_loss': ce_loss_value.item(),
            'rl_loss': rl_results['loss'].item(),
            'reward': rl_results['reward'],
            'baseline': rl_results['baseline'],
            'advantage': rl_results['advantage']
        }

