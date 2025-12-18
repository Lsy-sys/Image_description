"""
损失函数模块
包含交叉熵损失和SCST强化学习损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List


class CrossEntropyLoss(nn.Module):
    """交叉熵损失，支持标签平滑"""
    
    def __init__(self, ignore_index: int = 0, label_smoothing: float = 0.0):
        """
        Args:
            ignore_index: 忽略的索引（通常是padding）
            label_smoothing: 标签平滑系数
        """
        super().__init__()
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [batch_size, seq_len, vocab_size]
            targets: [batch_size, seq_len]
        Returns:
            loss: 标量
        """
        batch_size, seq_len, vocab_size = logits.shape
        
        # 展平
        logits_flat = logits.view(-1, vocab_size)
        targets_flat = targets.view(-1)
        
        if self.label_smoothing > 0:
            # 标签平滑
            log_probs = F.log_softmax(logits_flat, dim=1)
            with torch.no_grad():
                true_dist = torch.zeros_like(log_probs)
                true_dist.fill_(self.label_smoothing / (vocab_size - 2))
                true_dist.scatter_(1, targets_flat.unsqueeze(1), 1.0 - self.label_smoothing)
                true_dist[:, self.ignore_index] = 0
                mask = (targets_flat != self.ignore_index)
                true_dist = true_dist * mask.unsqueeze(1)
            
            loss = -torch.sum(true_dist * log_probs, dim=1)
            loss = loss.sum() / mask.sum()
        else:
            loss = F.cross_entropy(
                logits_flat, 
                targets_flat, 
                ignore_index=self.ignore_index,
                reduction='mean'
            )
        
        return loss


class SCSTLoss(nn.Module):
    """
    Self-Critical Sequence Training (SCST) 损失
    用于强化学习训练，直接优化评测指标
    """
    
    def __init__(
        self,
        reward_type: str = 'cider_d',
        baseline_type: str = 'self_critical',
        temperature: float = 1.0
    ):
        """
        Args:
            reward_type: 奖励类型 ('cider_d', 'bleu_4', 'rouge_l', 'combined')
            baseline_type: 基线类型 ('self_critical', 'average', 'none')
            temperature: 采样温度
        """
        super().__init__()
        self.reward_type = reward_type
        self.baseline_type = baseline_type
        self.temperature = temperature
        
    def forward(
        self,
        log_probs: torch.Tensor,
        sampled_seqs: torch.Tensor,
        ref_seqs: List[List[int]],
        vocab,
        greedy_seqs: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            log_probs: [batch_size, seq_len] 采样序列的对数概率
            sampled_seqs: [batch_size, seq_len] 采样的序列
            ref_seqs: List[List[int]] 参考序列
            vocab: 词汇表对象
            greedy_seqs: [batch_size, seq_len] 贪婪解码序列（用于self-critical baseline）
        
        Returns:
            Dict containing 'loss' and 'reward'
        """
        from evaluation.utils import compute_reward
        
        # 计算奖励
        rewards = compute_reward(
            sampled_seqs, 
            ref_seqs, 
            vocab, 
            reward_type=self.reward_type
        )  # [batch_size]
        
        # 计算基线
        if self.baseline_type == 'self_critical' and greedy_seqs is not None:
            baseline_rewards = compute_reward(
                greedy_seqs,
                ref_seqs,
                vocab,
                reward_type=self.reward_type
            )  # [batch_size]
            baseline = baseline_rewards
        elif self.baseline_type == 'average':
            baseline = rewards.mean()
        else:
            baseline = 0.0
        
        # 计算优势
        advantages = rewards - baseline
        
        # 计算损失（负的加权对数似然）
        loss = -(log_probs.sum(dim=1) * advantages).mean()
        
        return {
            'loss': loss,
            'reward': rewards.mean(),
            'baseline': baseline.mean() if isinstance(baseline, torch.Tensor) else baseline
        }









