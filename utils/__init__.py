"""
工具模块
包含日志记录、指标计算、可视化等工具
"""

from .logger import setup_logger
from .metrics import AverageMeter

# 延迟导入可视化模块，避免NumPy版本冲突
def plot_training_curves(*args, **kwargs):
    from .visualization import plot_training_curves
    return plot_training_curves(*args, **kwargs)

def visualize_attention(*args, **kwargs):
    from .visualization import visualize_attention
    return visualize_attention(*args, **kwargs)

__all__ = [
    'setup_logger',
    'AverageMeter',
    'plot_training_curves',
    'visualize_attention'
]
