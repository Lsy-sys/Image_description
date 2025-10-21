# 项目目录结构说明

## 整体结构

```
codes/
├── README.md                    # 项目说明文档
├── requirements.txt             # Python依赖包列表
├── test_structure.py           # 目录结构测试脚本
├── PROJECT_STRUCTURE.md        # 本文件
│
├── data/                       # 数据处理模块
│   ├── __init__.py
│   ├── dataset.py              # 数据集加载器
│   ├── transforms.py           # 图像变换
│   ├── vocabulary.py           # 词汇表构建
│   └── utils.py               # 数据工具函数
│
├── models/                     # 模型定义模块
│   ├── __init__.py
│   ├── cnn_gru/               # CNN+GRU模型
│   │   ├── __init__.py
│   │   ├── encoder.py         # CNN编码器
│   │   ├── decoder.py         # GRU解码器
│   │   └── model.py           # 完整模型
│   └── transformer/           # 区域特征+Transformer模型
│       ├── __init__.py
│       ├── attention.py       # 多头注意力机制
│       ├── encoder.py         # Transformer编码器
│       ├── decoder.py         # Transformer解码器
│       └── model.py           # 完整模型
│
├── evaluation/                # 评测指标模块
│   ├── __init__.py
│   ├── rouge_l.py            # ROUGE-L指标
│   ├── cider_d.py            # CIDEr-D指标
│   └── utils.py              # 评测工具函数
│
├── training/                  # 训练模块
│   ├── __init__.py
│   └── trainer.py            # 基础训练器
│
├── configs/                   # 配置文件
│   ├── cnn_gru_config.yaml   # CNN+GRU配置
│   ├── transformer_config.yaml # Transformer配置
│   └── rl_config.yaml        # 强化学习配置
│
├── utils/                     # 通用工具
│   ├── __init__.py
│   ├── logger.py             # 日志记录
│   ├── metrics.py            # 指标计算
│   └── visualization.py      # 可视化工具
│
├── scripts/                   # 运行脚本
│   ├── train_cnn_gru.py      # 训练CNN+GRU模型
│   ├── evaluate.py           # 模型评测
│   ├── download_and_prepare_data.py # 数据下载和预处理
│   └── verify_data.py        # 数据验证
│
└── notebooks/                 # Jupyter笔记本
    └── data_analysis.ipynb   # 数据分析笔记本
```

## 主要功能模块

### 1. 数据处理 (data/)
- **dataset.py**: 实现DeepFashion-MultiModal数据集加载器
- **transforms.py**: 图像预处理和变换
- **vocabulary.py**: 词汇表构建和管理
- **utils.py**: 数据加载和批处理工具

### 2. 模型实现 (models/)
- **CNN+GRU模型**: 基线模型，使用ResNet提取全局特征，GRU生成描述
- **区域特征+Transformer模型**: 先进模型，使用Faster R-CNN提取区域特征，Transformer编解码

### 3. 评测指标 (evaluation/)
- **ROUGE-L**: 基于最长公共子序列的评测指标
- **CIDEr-D**: 基于TF-IDF的n-gram相似度评测指标

### 4. 训练模块 (training/)
- **基础训练器**: 提供通用的训练框架
- **模型特定训练器**: 针对不同模型的训练逻辑
- **强化学习训练器**: 实现基于CIDEr-D的强化学习优化

### 5. 配置和工具
- **配置文件**: YAML格式的模型和训练配置
- **工具函数**: 日志记录、指标计算、可视化等

## 使用说明

### 1. 环境设置
```bash
pip install -r requirements.txt
```

### 2. 训练模型
```bash
# 训练CNN+GRU模型
python scripts/train_cnn_gru.py --config configs/cnn_gru_config.yaml

# 训练Transformer模型
python scripts/train_transformer.py --config configs/transformer_config.yaml

# 强化学习训练
python scripts/train_rl.py --config configs/rl_config.yaml
```

### 3. 模型评测
```bash
python scripts/evaluate.py --model_type cnn_gru --config configs/cnn_gru_config.yaml --checkpoint checkpoints/cnn_gru/best_model.pth
```

### 4. 数据分析
打开 `notebooks/data_analysis.ipynb` 进行数据探索和分析。

## 项目特点

1. **模块化设计**: 各功能模块独立，便于维护和扩展
2. **配置驱动**: 使用YAML配置文件管理超参数
3. **完整流程**: 从数据处理到模型训练再到评测的完整流程
4. **可扩展性**: 易于添加新的模型架构和评测指标
5. **文档完善**: 详细的代码注释和使用说明

## 注意事项

1. 确保数据目录结构正确
2. 根据硬件配置调整批次大小
3. 定期保存检查点避免训练中断
4. 使用GPU加速训练过程
