"""
优化器和学习率调度器工厂
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import (
    ReduceLROnPlateau,
    CosineAnnealingLR,
    StepLR,
    LambdaLR
)
from typing import Dict, Any, Optional


def create_optimizer(
    model: torch.nn.Module,
    config: Dict[str, Any]
) -> torch.optim.Optimizer:
    """
    根据配置创建优化器
    
    Args:
        model: 模型
        config: 优化器配置字典
            - type: 'adam', 'adamw', 'sgd'
            - lr: 学习率
            - weight_decay: 权重衰减
            - betas: (for Adam) [0.9, 0.999]
            - momentum: (for SGD) 0.9
    """
    # 支持两类配置键名：'type' 或向后兼容的 'optimizer'
    if isinstance(config, dict):
        optimizer_type = str(config.get('type') or config.get('optimizer') or 'adam').lower()
        # 支持 'lr' 或向后兼容的 'learning_rate'
        lr = float(config.get('lr') or config.get('learning_rate') or 0.001)
        weight_decay = float(config.get('weight_decay', 0.0))
    else:
        # 如果传入的是字符串或其他类型，退回到默认
        optimizer_type = str(config).lower() if config else 'adam'
        lr = 0.001
        weight_decay = 0.0
    
    params = [p for p in model.parameters() if p.requires_grad]
    
    if optimizer_type == 'adam':
        betas = config.get('betas', [0.9, 0.999])
        optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay, betas=betas)
    elif optimizer_type == 'adamw':
        betas = config.get('betas', [0.9, 0.999])
        optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas)
    elif optimizer_type == 'sgd':
        momentum = config.get('momentum', 0.9)
        optimizer = optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=momentum)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    return optimizer


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Dict[str, Any],
    num_training_steps: Optional[int] = None
):
    """
    根据配置创建学习率调度器
    
    Args:
        optimizer: 优化器
        config: 调度器配置字典
            - type: 'plateau', 'cosine', 'step', 'warmup_cosine'
            - mode: (for plateau) 'min' or 'max'
            - factor: (for plateau/step) 衰减因子
            - patience: (for plateau) 耐心值
            - min_lr: (for plateau) 最小学习率
            - T_max: (for cosine) 最大步数
            - step_size: (for step) 步长
            - gamma: (for step) 衰减率
            - warmup_steps: (for warmup_cosine) warmup步数
        num_training_steps: 总训练步数（用于cosine调度器）
    """
    scheduler_type = config.get('type', 'plateau').lower()
    
    if scheduler_type == 'plateau':
        mode = config.get('mode', 'min')
        factor = config.get('factor', 0.5)
        patience = config.get('patience', 5)
        min_lr = config.get('min_lr', 1e-6)
        scheduler = ReduceLROnPlateau(
            optimizer, mode=mode, factor=factor, 
            patience=patience, min_lr=min_lr, verbose=True
        )
    elif scheduler_type == 'cosine':
        T_max = config.get('T_max', num_training_steps or 10000)
        eta_min = config.get('eta_min', 0)
        scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    elif scheduler_type == 'step':
        step_size = config.get('step_size', 10)
        gamma = config.get('gamma', 0.1)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == 'warmup_cosine':
        # Warmup + Cosine Annealing
        warmup_steps = config.get('warmup_steps', 1000)
        T_max = config.get('T_max', num_training_steps or 10000)

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, T_max - warmup_steps))
            return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.141592653589793))))

        scheduler = LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = None

    return scheduler



