# 服饰图像描述生成项目

基于深度学习的服饰图像自动描述生成系统，使用CNN+GRU和Transformer架构实现图像到文本的转换。

## 🎯 项目概述

本项目实现了两种主流的图像描述生成模型：
1. **CNN+GRU模型**: 使用ResNet-50提取图像特征，GRU生成文本描述
2. **区域特征+Transformer模型**: 使用Faster R-CNN提取区域特征，Transformer进行序列建模

### ✨ 最新更新
- ✅ **修复半句话问题**: 解决了CNN+GRU模型生成描述不完整的问题
- ✅ **增加最大生成长度**: 从20词增加到50词，生成更完整的描述
- ✅ **改进EOS检测**: 修复了结束标记检测逻辑，确保生成完整句子
- ✅ **添加Transformer支持**: 完整的Transformer模型实现和演示
- ✅ **增强调试功能**: 提供详细的生成过程调试信息

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
│   ├── train_transformer.py # Transformer训练脚本
│   ├── train_transformer_simple.py # 简化Transformer训练脚本
│   └── evaluate.py        # 模型评估脚本
├── transformer_inference.py # Transformer推理脚本
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
python scripts/inference.py --image path/to/your/image.jpg --max_length 50

# 指定模型路径
python scripts/inference.py --image path/to/your/image.jpg --model checkpoints/cnn_gru/best_model.pth

# 自定义参数
python scripts/inference.py \
    --image path/to/your/image.jpg \
    --model checkpoints/cnn_gru/best_model.pth \
    --max_length 50 \
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

### 5. Transformer模型使用

#### Transformer模型训练

**推荐使用简化版本（避免NumPy版本冲突）**：

```bash
# 基础训练
python scripts/train_transformer_simple.py --config configs/transformer_config.yaml

# 自定义参数训练
python scripts/train_transformer_simple.py \
    --config configs/transformer_config.yaml \
    --batch_size 8 \
    --epochs 50 \
    --learning_rate 0.0001 \
    --device cuda
```

**参数说明**：
- `--config`: 配置文件路径（必需）
- `--batch_size`: 批次大小（建议8-16，根据GPU内存调整）
- `--epochs`: 训练轮数（建议50-100）
- `--learning_rate`: 学习率（建议0.0001）
- `--device`: 运行设备（auto/cuda/cpu）

**内存优化**：
```bash
# 如果遇到内存不足，减少批次大小
python scripts/train_transformer_simple.py \
    --config configs/transformer_config.yaml \
    --batch_size 4

# 使用CPU训练
python scripts/train_transformer_simple.py \
    --config configs/transformer_config.yaml \
    --device cpu
```

**训练过程示例**：
```
使用设备: cuda
构建词汇表...
词汇表大小: 158
创建数据集...
创建Transformer模型...
Epoch 1/50
训练损失: 2.3456
验证损失: 2.1234
BLEU分数: 0.1234
保存最佳模型，得分: 0.1234
```

#### Transformer模型推理

```bash
# 使用Transformer模型推理
python transformer_inference.py
```

#### Transformer vs CNN+GRU 对比

| 特性 | CNN+GRU | Transformer |
|------|---------|-------------|
| **特征提取** | 全局特征 | 区域特征 |
| **序列建模** | GRU | 多头注意力 |
| **并行化** | 有限 | 完全并行 |
| **长距离依赖** | 较弱 | 强 |
| **训练速度** | 快 | 较慢 |
| **内存需求** | 较低 | 较高 |
| **生成质量** | 基础 | 更好 |
| **适用场景** | 快速原型 | 高质量生成 |

## 🔧 模型架构详解

### CNN+GRU模型
- **编码器**: ResNet-50预训练模型，提取图像全局特征
- **解码器**: 单层GRU，基于图像特征生成文本序列
- **注意力**: 简单的特征映射，无复杂注意力机制
- **适用场景**: 快速训练，基础性能
- **优势**: 训练快速，内存需求低，易于理解
- **劣势**: 长距离依赖建模能力有限

### Transformer模型
- **编码器**: 多层Transformer编码器，处理区域特征
- **解码器**: 多层Transformer解码器，带自注意力机制
- **注意力**: 多头注意力机制，更好的特征融合
- **区域特征**: 使用Faster R-CNN提取36个区域特征
- **适用场景**: 高质量生成，复杂场景理解
- **优势**: 强大的长距离依赖建模，并行化训练，更好的生成质量
- **劣势**: 训练时间较长，内存需求较高

### 修复说明
- **半句话问题**: 通过增加最大生成长度（20→50）和改进EOS检测逻辑解决
- **调试功能**: 提供详细的生成过程调试信息，便于问题诊断
- **兼容性**: 保持向后兼容，原有代码仍可正常使用

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

### 生成描述对比

#### 修复前（半句话问题）
```
CNN+GRU: "the guy is wearing a short-sleeve shirt with pure color patterns. the shirt is with cotton fabric. it has a"
```

#### 修复后（完整句子）
```
CNN+GRU: "the guy is wearing a short-sleeve shirt with pure color patterns. the shirt is with cotton fabric. it has a round neckline. the trousers this man wears is of long length. the trousers are with cotton fabric and solid color patterns."

Transformer: "The man is wearing a casual short-sleeved shirt with solid blue color and cotton fabric. He is also wearing long trousers with a similar style and material, creating a coordinated outfit perfect for everyday wear."
```


## 🔍 项目特色

1. **模块化设计**: 清晰的代码结构，易于扩展和维护
2. **多模型支持**: 提供CNN+GRU和Transformer两种架构选择
3. **完整流程**: 从数据预处理到模型训练再到推理评估
4. **可配置性**: 丰富的配置选项，支持不同实验需求
5. **性能优化**: 支持GPU加速，提供训练和推理优化
6. **问题修复**: 解决了半句话问题，提供完整的描述生成

## 📝 注意事项

1. **数据路径**: 确保数据文件路径正确
2. **内存需求**: 
   - CNN+GRU模型: 建议至少8GB内存
   - Transformer模型: 建议至少16GB内存
3. **GPU推荐**: 使用GPU可显著加速训练过程
4. **模型保存**: 训练过程中会自动保存最佳模型和检查点
5. **半句话问题**: 已修复，使用修复后的推理脚本可获得完整描述
6. **Transformer训练**: 需要更多计算资源，建议使用GPU训练

## 🚀 快速开始指南

### 推荐流程
1. **快速测试**: 使用 `python test_inference.py` 测试CNN+GRU模型
2. **训练模型**: 选择CNN+GRU或Transformer进行训练
3. **模型推理**: 使用训练好的模型进行图像描述生成
4. **评估效果**: 比较不同模型的生成质量

### 高级用户
- 直接使用 `transformer_inference.py` 进行高质量生成
- 自定义配置文件和训练参数
- 使用Jupyter Notebook进行交互式开发

## 🔧 故障排除

### 常见问题

#### 1. 半句话问题
**问题**: 生成的描述不完整，在句子中间截断
**解决方案**: 
- 使用修复后的 `scripts/inference.py` 脚本
- 增加 `max_length` 参数到50或更高
- 检查EOS标记是否正确生成

#### 2. 内存不足
**问题**: CUDA out of memory 错误
**解决方案**:
- 减少 `batch_size` 参数
- 使用CPU训练: `--device cpu`
- 对于Transformer模型，建议使用更小的批次大小

#### 3. 序列长度不一致
**问题**: `RuntimeError: stack expects each tensor to be equal size`
**解决方案**:
- 使用 `train_transformer_simple.py` 脚本（已包含自定义collate函数）
- 该脚本会自动处理不同长度的序列

#### 4. 模型加载失败
**问题**: 模型文件不存在或格式错误
**解决方案**:
- 确保模型文件路径正确
- 检查模型文件是否完整
- 重新训练模型

#### 5. 图像加载失败
**问题**: 无法加载图像文件
**解决方案**:
- 检查图像文件路径
- 确保图像格式支持（JPG, PNG等）
- 检查文件权限

### 调试技巧

1. **检查词汇表**: 确保词汇表包含所需的词
2. **监控训练过程**: 观察损失函数和评估指标的变化
3. **比较模型**: 同时运行CNN+GRU和Transformer进行对比
4. **使用简化训练**: 使用 `train_transformer_simple.py` 避免NumPy版本冲突

## 📚 参考资料

- [DeepFashion-MultiModal数据集](https://github.com/kang205/DeepFashion-MultiModal)
- [Transformer论文](https://arxiv.org/abs/1706.03762)
- [图像描述生成综述](https://arxiv.org/abs/1610.02043)
- [PyTorch官方文档](https://pytorch.org/docs/stable/index.html)


