# 服饰图像描述生成项目

基于深度学习的服饰图像自动描述生成系统，使用CNN+GRU和Transformer架构实现图像到文本的转换。

## 🎯 项目概述

本项目实现了两种主流的图像描述生成模型：
1. **CNN+GRU模型**: 使用ResNet-50提取图像特征，GRU生成文本描述
2. **区域特征+Transformer模型**: 使用Faster R-CNN提取区域特征，Transformer进行序列建模

### 主要功能
- 🖼️ **图像理解**: 自动分析服饰图像中的颜色、款式、材质等特征
- 📝 **文本生成**: 生成自然、准确的服饰描述文本
- 🔄 **多模型支持**: 提供多种模型架构选择
- 📊 **性能评估**: 使用ROUGE-L和CIDEr-D指标评估生成质量

## 📁 项目结构

```
codes/
├── data/                    # 数据处理模块
│   ├── dataset.py          # DeepFashion数据集加载器
│   ├── transforms.py       # 图像预处理和增强
│   ├── vocabulary.py       # 词汇表构建和管理
│   └── utils.py           # 数据工具函数
├── models/                 # 模型定义
│   ├── cnn_gru/           # CNN+GRU模型
│   │   ├── encoder.py     # ResNet-50图像编码器
│   │   ├── decoder.py     # GRU文本解码器
│   │   └── model.py       # 完整CNN+GRU模型
│   └── transformer/       # Transformer模型
│       ├── encoder.py     # Transformer编码器
│       ├── decoder.py     # Transformer解码器
│       ├── attention.py   # 多头注意力机制
│       └── model.py       # 完整Transformer模型
├── evaluation/            # 评测指标
│   ├── rouge_l.py        # ROUGE-L评测指标
│   └── cider_d.py        # CIDEr-D评测指标
├── training/              # 训练模块
│   └── trainer.py        # 训练器基类
├── configs/               # 配置文件
│   ├── cnn_gru_config.yaml      # CNN+GRU模型配置
│   └── transformer_config.yaml  # Transformer模型配置
├── scripts/               # 运行脚本
│   ├── simple_train.py    # 简化训练脚本
│   ├── train_cnn_gru.py   # CNN+GRU训练脚本
│   └── evaluate.py        # 模型评估脚本
├── checkpoints/           # 模型检查点
├── logs/                  # 训练日志
└── requirements.txt       # 依赖包列表
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install -r requirements.txt

# 确保CUDA可用（可选，用于GPU加速）
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

### 2. 数据准备

项目使用DeepFashion-MultiModal数据集，包含：
- **图像数据**: 44,096张服饰图像
- **文本描述**: 每个图像对应多个自然语言描述
- **数据分割**: 训练集(29,780) + 验证集(6,381) + 测试集(8,000)

#### 数据获取方式

由于数据集图片文件较大，图片文件不会包含在Git仓库中。请按以下方式获取：

1. **下载DeepFashion-MultiModal数据集**到 `data/DeepFashion-MultiModal/` 目录
2. **确保目录结构如下**：
```
data/DeepFashion-MultiModal/
├── images/           # 所有图片文件（需要下载）
├── captions/         # 所有标注文件（需要下载）
├── train_list.txt    # 训练集列表
├── val_list.txt      # 验证集列表
└── test_list.txt     # 测试集列表
```

3. **下载预训练模型**到 `checkpoints/cnn_gru/best_model.pth`（可选，用于快速测试）

### 3. 模型训练

#### CNN+GRU模型训练
```bash
# 基础训练
python scripts/simple_train.py --config configs/cnn_gru_config.yaml --batch_size 16

# 自定义参数训练
python scripts/simple_train.py \
    --config configs/cnn_gru_config.yaml \
    --batch_size 32 \
    --epochs 100 \
    --learning_rate 0.0001
```

#### 训练参数说明
- `--config`: 配置文件路径
- `--batch_size`: 批次大小（建议GPU: 16-32, CPU: 8-16）
- `--epochs`: 训练轮数（默认50）
- `--learning_rate`: 学习率（默认0.001）

### 4. 模型推理

训练完成后，模型会保存在 `checkpoints/cnn_gru/best_model.pth`。

#### 方法1: 命令行推理 (推荐)

```bash
# 基础使用
python scripts/inference.py --image path/to/your/image.jpg

# 指定模型路径
python scripts/inference.py --image path/to/your/image.jpg --model checkpoints/cnn_gru/best_model.pth

# 自定义参数
python scripts/inference.py \
    --image path/to/your/image.jpg \
    --model checkpoints/cnn_gru/best_model.pth \
    --max_length 30 \
    --device cuda
```

#### 方法2: Python代码使用

```python
# 使用演示脚本
python scripts/demo.py

# 或在代码中直接使用
import torch
from models.cnn_gru.model import CNNGruModel
from data.transforms import ImageTransforms
from PIL import Image

# 加载模型
checkpoint = torch.load('checkpoints/cnn_gru/best_model.pth')
vocab = checkpoint['vocab']

model = CNNGruModel(
    embed_size=512,
    hidden_size=512,
    vocab_size=len(vocab),
    num_layers=1,
    pretrained=True
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 图像预处理
transform = ImageTransforms(224, is_training=False)
image = Image.open('path/to/image.jpg').convert('RGB')
image_tensor = transform.get_transforms()(image).unsqueeze(0)

# 生成描述
def generate_caption(image_tensor, model, vocab, max_length=20):
    with torch.no_grad():
        # 提取图像特征
        features = model.encoder(image_tensor)
        
        # 生成文本
        caption = []
        hidden = model.decoder.init_hidden(1)
        
        for _ in range(max_length):
            output, hidden = model.decoder(features, hidden)
            word_id = output.argmax(1)
            
            if word_id.item() == vocab.word2idx['<eos>']:
                break
                
            word = vocab.idx2word[word_id.item()]
            if word not in ['<bos>', '<pad>', '<unk>']:
                caption.append(word)
        
        return ' '.join(caption)

# 生成描述
description = generate_caption(image_tensor, model, vocab)
print(f"生成的描述: {description}")
```

#### 方法3: Jupyter Notebook使用

```bash
# 启动Jupyter Notebook
jupyter notebook notebooks/image_caption_demo.ipynb
```

#### 方法4: 批量处理

```python
from scripts.demo import ImageCaptionGenerator

# 创建描述生成器
generator = ImageCaptionGenerator('checkpoints/cnn_gru/best_model.pth')

# 单张图像
caption = generator.generate_caption('path/to/image.jpg')
print(f"描述: {caption}")

# 批量处理
image_paths = ['image1.jpg', 'image2.jpg', 'image3.jpg']
results = generator.batch_generate(image_paths)
for path, caption in results.items():
    print(f"{path}: {caption}")
```

## 🔧 模型架构详解

### CNN+GRU模型
- **编码器**: ResNet-50预训练模型，提取图像全局特征
- **解码器**: 单层GRU，基于图像特征生成文本序列
- **注意力**: 简单的特征映射，无复杂注意力机制
- **适用场景**: 快速训练，基础性能

### Transformer模型
- **编码器**: 多层Transformer编码器，处理区域特征
- **解码器**: 多层Transformer解码器，带自注意力机制
- **注意力**: 多头注意力机制，更好的特征融合
- **适用场景**: 高质量生成，复杂场景理解

## 📊 性能评估

### 评测指标
- **ROUGE-L**: 基于最长公共子序列，评估生成文本的流畅性
- **CIDEr-D**: 基于TF-IDF的n-gram相似度，评估描述的准确性

### 运行评估
```bash
python scripts/evaluate.py \
    --model_path checkpoints/cnn_gru/best_model.pth \
    --test_data data/DeepFashion-MultiModal/test_list.txt
```

## ⚙️ 配置说明

### CNN+GRU配置 (configs/cnn_gru_config.yaml)
```yaml
model:
  embed_size: 512        # 嵌入维度
  hidden_size: 512       # GRU隐藏层大小
  vocab_size: 10000      # 词汇表大小
  num_layers: 1          # GRU层数
  dropout: 0.5           # Dropout率

training:
  epochs: 50             # 训练轮数
  batch_size: 32         # 批次大小
  learning_rate: 0.001   # 学习率
  optimizer: 'adam'      # 优化器
  grad_clip: 5.0         # 梯度裁剪

data:
  image_size: 224        # 图像尺寸
  max_caption_length: 20 # 最大描述长度
  min_freq: 5            # 最小词频
```

## 🎨 使用示例

### 输入图像
![示例服饰图像](示例图像路径)

### 生成描述
- **CNN+GRU**: "A woman wearing a blue dress with floral patterns"
- **Transformer**: "The woman is wearing a sleeveless blue dress with white floral patterns, perfect for summer occasions"

## 🔍 项目特色

1. **模块化设计**: 清晰的代码结构，易于扩展和维护
2. **多模型支持**: 提供多种模型架构选择
3. **完整流程**: 从数据预处理到模型训练再到推理评估
4. **可配置性**: 丰富的配置选项，支持不同实验需求
5. **性能优化**: 支持GPU加速，提供训练和推理优化

## 📝 注意事项

1. **数据路径**: 确保数据文件路径正确
2. **内存需求**: 建议至少8GB内存，推荐16GB以上
3. **GPU推荐**: 使用GPU可显著加速训练过程
4. **模型保存**: 训练过程中会自动保存最佳模型和检查点


