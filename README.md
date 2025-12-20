# DeepFashion 图像描述生成系统

基于深度学习的服饰图像自动描述生成系统，采用**配置驱动、模块化、工业级**架构设计。

## 🎯 项目特点

- **配置驱动架构**：所有模型通过YAML配置文件动态组装
- **模块化设计**：Encoder/Decoder完全分离，易于扩展
- **5个模型架构**：从基线到前沿的完整对比
- **5个核心实验**：完整的科研实验设计
- **统一接口**：所有模型使用相同的接口和训练流程

## 📁 项目结构

```
DeepFashion-Captioning/
├── configs/                    # 配置中心
│   ├── base.yaml               # 基础配置（路径、数据、训练通用参数）
│   ├── models/                 # 模型配置
│   │   ├── 1_cnn_gru.yaml      # Model A: ResNet + GRU
│   │   ├── 2_attn_gru.yaml     # Model B: ResNet + Attention + GRU
│   │   ├── 3_region_trans.yaml # Model C: FasterRCNN + Transformer
│   │   ├── 4_vit_trans.yaml    # Model D: ViT + Transformer
│   │   └── 5_graph_trans.yaml  # Model E: GCN + Transformer
│   └── strategies/             # 训练策略配置
│       ├── xe_training.yaml    # 交叉熵训练
│       └── rl_finetune.yaml    # 强化学习微调
├── models/                     # 模型层
│   ├── __init__.py            # Model Factory（根据配置创建模型）
│   ├── captioner.py           # 统一接口（ImageCaptioner）
│   ├── encoders/              # 视觉编码器
│   │   ├── cnn_encoder.py     # ResNetEncoder
│   │   ├── region_encoder.py  # FasterRCNNEncoder
│   │   ├── vit_encoder.py     # ViTEncoder
│   │   └── graph_encoder.py   # GCNEncoder
│   ├── decoders/              # 文本解码器
│   │   ├── rnn_decoder.py     # GRUDecoder, AttnGRUDecoder
│   │   └── trans_decoder.py   # TransformerDecoder
│   └── layers/                # 公共层
│       ├── attention.py       # MultiHeadAttention, PositionalEncoding
│       └── embeddings.py     # WordEmbedding
├── modules/                    # 组件层
│   ├── losses.py              # CrossEntropyLoss, SCSTLoss
│   ├── optimizers.py          # 优化器和调度器工厂
│   └── metrics.py             # 统一指标计算接口
├── data/                       # 数据层
│   ├── dataset.py             # DeepFashionDataset
│   ├── vocabulary.py         # Vocabulary
│   ├── transforms.py         # ImageTransforms
│   └── utils.py              # 数据工具函数
├── evaluation/                 # 评测指标
│   ├── bleu.py               # BLEU指标
│   ├── cider_d.py            # CIDEr-D指标
│   ├── rouge_l.py            # ROUGE-L指标
│   └── utils.py              # 评测工具
├── training/                   # 训练模块
│   └── trainer.py            # BaseTrainer
├── analysis/                   # 分析脚本（5个实验）
│   ├── plot_performance.py   # 实验一：模型对比
│   ├── visualize_attention.py # 实验二：注意力可视化
│   ├── analyze_length.py     # 实验三：长度敏感度
│   ├── analyze_diversity.py  # 实验四：解码策略
│   └── plot_rl_dynamics.py   # 实验五：RL分析
├── scripts/                    # 执行脚本
│   ├── train.py              # 统一训练入口
│   ├── inference.py           # 推理入口
│   ├── evaluate.py            # 评估入口
│   └── ...                   # 其他工具脚本
├── api/                        # API服务
│   └── server.py             # FastAPI服务
├── web/                        # 前端（可视化工作台）
│   ├── index.html             # DeepFashion AI Workbench 主页面 + 导航
│   ├── styles.css             # 赛博朋克暗色系UI
│   └── js/main.js             # 前端交互逻辑（模型选择、X-Ray、Training Monitor等）
└── checkpoints/                 # 模型检查点
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install -r requirements.txt

# 检查CUDA（可选）
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

### 2. 数据准备

项目使用DeepFashion-MultiModal数据集，确保数据目录结构如下：

```
data/DeepFashion-MultiModal/
├── images/           # 图像文件
├── captions/         # 标注文件
├── train_list.txt    # 训练集列表
├── val_list.txt      # 验证集列表
└── test_list.txt     # 测试集列表
```

### 3. 训练模型

```bash
# Model A: CNN+GRU (基线模型)
python scripts/train.py --config configs/models/1_cnn_gru.yaml

# Model B: Attn-GRU (带空间注意力)
python scripts/train.py --config configs/models/2_attn_gru.yaml

# Model C: Region-Transformer (主力模型)
python scripts/train.py --config configs/models/3_region_trans.yaml

# Model D: ViT-Transformer (Vision Transformer)
python scripts/train.py --config configs/models/4_vit_trans.yaml

# Model E: Graph-Transformer (图卷积网络)
# 采用 ViT patch tokens 作为图节点特征（推荐、最省事，不依赖检测器）
# 约定：
# - node_features: (196, 768)  # 14x14 patches
# - adj_matrix:    (196, 196)  # patch 网格邻接
# - 默认特征目录：data/vit_features（已在 configs/base.yaml 配好）
#
# 1) 预提取 ViT patch 节点特征（保存到 data/vit_features）
python data/preprocess_vit_patches.py --data_dir data/DeepFashion-MultiModal --output_dir data/vit_features --split all

# 2) 构建 patch 网格图结构（保存到 data/graphs）
python data/build_graphs.py --data_dir data/DeepFashion-MultiModal --output_dir data/graphs --graph_type spatial --split all --node_type vit_patches

# 3) 使用 GCN + Transformer 训练 Model E
python scripts/train.py --config configs/models/5_graph_trans.yaml

# 使用RL微调（需要先完成XE训练）
# 推荐在Model C上使用RL微调
python scripts/train.py --config configs/models/3_region_trans.yaml --strategy rl_finetune
```

### 4. 模型推理

```bash
# 使用训练好的模型进行推理
# Model A: CNN+GRU (基线模型)
python scripts/inference.py --image data\DeepFashion-MultiModal\images\MEN-Denim-id_00000080-01_7_additional.jpg --config configs/models/1_cnn_gru.yaml --checkpoint checkpoints/cnn_gru/best_model.pth

# Model B: Attn-GRU (带空间注意力)
python scripts/inference.py --image data\DeepFashion-MultiModal\images\MEN-Denim-id_00000080-01_7_additional.jpg --config configs/models/2_attn_gru.yaml --checkpoint checkpoints/attn_gru/best_model.pth

# Model C: Region-Transformer (主力模型)
python scripts/inference.py --image data\DeepFashion-MultiModal\images\MEN-Denim-id_00000080-01_7_additional.jpg --config configs/models/3_region_trans.yaml --checkpoint checkpoints/region_transformer/best_model.pth

# Model D: ViT-Transformer (Vision Transformer)
python scripts/inference.py --image data\DeepFashion-MultiModal\images\MEN-Denim-id_00000080-01_7_additional.jpg --config configs/models/4_vit_trans.yaml --checkpoint checkpoints/vit_transformer/best_model.pth


# Model E（Graph-Transformer）使用 item_id 推理（读取 data/vit_features 与 data/graphs 的 .npy）
python scripts/inference.py --config configs/models/5_graph_trans.yaml --item_id "WOMEN-Tees_Tanks-id_00003315-01_3_back"
```

### 5. 模型评估

```bash
# 评估模型性能
# Model A: CNN+GRU (基线模型)
python scripts/evaluate.py --config configs/models/1_cnn_gru.yaml --checkpoint checkpoints/cnn_gru/best_model.pth
# Model B: Attn-GRU (带空间注意力)
python scripts/evaluate.py --config configs/models/2_attn_gru.yaml --checkpoint checkpoints/attn_gru/best_model.pth
# Model C: Region-Transformer (主力模型)
python scripts/evaluate.py --config configs/models/3_region_trans.yaml --checkpoint checkpoints/region_transformer/best_model.pth
# Model D: ViT-Transformer (Vision Transformer)
python scripts/evaluate.py --config configs/models/4_vit_trans.yaml --checkpoint checkpoints/vit_transformer/best_model.pth

# Model E（Graph-Transformer）评估
python scripts/evaluate.py --config configs/models/5_graph_trans.yaml --item_id "WOMEN-Tees_Tanks-id_00003315-01_3_back"
```

### 6. 启动后端与前端可视化

```bash
# 启动FastAPI后端（提供 /api/predict、/api/attention、/api/training_log 等接口）
uvicorn api.server:app --reload --host 0.0.0.0 --port 8000

# 在本地启动前端静态服务器（推荐）
cd web
python -m http.server 5500
```

然后在浏览器中访问：

- 工作台页面：`http://127.0.0.1:5500/index.html`
- 后端Swagger文档：`http://127.0.0.1:8000/docs`

## 📊 五大核心实验

### 实验一：模型架构横向对比
```bash
python analysis/plot_performance.py
```
- 对比5个模型的Loss收敛曲线
- 绘制综合性能雷达图（BLEU-4, METEOR, ROUGE-L, CIDEr-D, SPICE）

### 实验二：注意力机制可视化
```bash
python analysis/visualize_attention.py
```
- 可视化不同模型的注意力权重
- 对比Spatial Attention vs Self-Attention

### 实验三：序列长度敏感度分析
```bash
python analysis/analyze_length.py
```
- 分析模型在不同长度区间的性能
- 验证Transformer的长文本建模能力

### 实验四：解码策略对比
```bash
python analysis/analyze_diversity.py
```
- 对比Greedy、Beam Search、Sampling策略
- 分析词频分布和多样性指标

### 实验五：RL优化目标分析
```bash
python analysis/plot_rl_dynamics.py
```
- 对比XE vs RL训练
- 分析CIDEr vs BLEU作为奖励的效果
- 依赖训练产生的 `training_log.json` 与参考语料进行曲线与TF-IDF分析

> 说明：训练期间，`BaseTrainer` 会在每个epoch结束后自动将
> `epochs / train_loss / val_loss` 写入对应模型日志目录（例如 `logs/region_transformer/training_log.json`），
> 前端的 **Training Monitor** 与本脚本都会复用这份日志。

## ⚙️ 配置说明

### 模型配置示例

```yaml
# configs/models/1_cnn_gru.yaml
_base_: '../base.yaml'  # 继承基础配置

model:
  type: 'cnn_gru'
  encoder:
    type: 'resnet'
    embed_size: 512
    pretrained: true
  decoder:
    type: 'gru'
    hidden_size: 512
    num_layers: 1
    dropout: 0.5

training:
  epochs: 30
  batch_size: 32
  learning_rate: 0.001
```

### 训练策略配置

```yaml
# configs/strategies/xe_training.yaml
loss:
  type: 'cross_entropy'
  label_smoothing: 0.1

optimizer:
  type: 'adam'
  lr: 0.001
  weight_decay: 1e-4

scheduler:
  type: 'plateau'
  mode: 'min'
  patience: 5
```

## 🔧 核心设计理念

1. **配置驱动**：所有模型通过配置文件定义，无需修改代码
2. **模块化组装**：Encoder和Decoder独立，可自由组合
3. **统一接口**：所有模型使用`ImageCaptioner`统一接口
4. **组件化**：Losses、Optimizers、Metrics独立模块
5. **可扩展**：易于添加新模型、新实验、新功能

## 🖥️ 前端 Workbench 功能概览

- **Workbench 主页面**：上传服饰图片、选择5种模型架构、切换解码策略（Greedy / Beam / Sampling），实时查看生成描述与 BLEU / CIDEr / ROUGE 指标动画。
- **X-Ray Vision 页面**：点击生成句子中的单词，在原图上叠加注意力热力图，实现“单词级聚焦区域”可视化（通过 `/api/attention` 获取热力图）。
- **Training Monitor 页面**：基于 `training_log.json` 绘制 XE / RL 训练损失随 Epoch 变化的折线图，支持鼠标悬停查看精确数值。

## 📝 使用Model Factory

```python
from models import create_model_from_config, load_config

# 从配置文件创建模型
config = load_config('configs/models/1_cnn_gru.yaml')
model = create_model_from_config('configs/models/1_cnn_gru.yaml', vocab_size=10000)
```

### 模型定义位置和方式

五个模型采用**配置驱动 + 模块化组装**的方式定义：

#### 定义位置总览

```
models/
├── __init__.py              # Model Factory（模型工厂）
├── captioner.py             # 统一接口（ImageCaptioner）
├── encoders/                # 视觉编码器定义
│   ├── cnn_encoder.py      # ResNetEncoder (Model A, B)
│   ├── region_encoder.py   # FasterRCNNEncoder (Model C)
│   ├── vit_encoder.py       # ViTEncoder (Model D)
│   └── graph_encoder.py     # GCNEncoder (Model E)
└── decoders/                # 文本解码器定义
    ├── rnn_decoder.py       # GRUDecoder, AttnGRUDecoder (Model A, B)
    └── trans_decoder.py     # TransformerDecoder (Model C, D, E)

configs/models/              # 模型配置文件
├── 1_cnn_gru.yaml          # Model A配置
├── 2_attn_gru.yaml         # Model B配置
├── 3_region_trans.yaml     # Model C配置
├── 4_vit_trans.yaml        # Model D配置
└── 5_graph_trans.yaml      # Model E配置
```

#### 模型定义流程

1. **配置文件定义（YAML）**：每个模型通过YAML配置文件定义其架构和参数
2. **Model Factory 组装**：`models/__init__.py` 中的 `create_model()` 函数根据配置动态组装模型
3. **统一接口封装**：`ImageCaptioner` 类提供统一接口，连接编码器和解码器

#### 模型定义流程

1. **配置文件定义（YAML）**：每个模型通过YAML配置文件定义其架构和参数
2. **Model Factory 组装**：`models/__init__.py` 中的 `create_model()` 函数根据配置动态组装模型
3. **统一接口封装**：`ImageCaptioner` 类提供统一接口，连接编码器和解码器

#### 模型创建流程

```
配置文件 (YAML)
    ↓
Model Factory 解析 (models/__init__.py)
    ↓
动态导入对应的 Encoder 和 Decoder 类
    ↓
根据配置参数实例化 Encoder
    ↓
根据配置参数实例化 Decoder
    ↓
ImageCaptioner 封装 (encoder, decoder)
    ↓
返回完整模型实例
```

#### 各模型详细定义位置

- **Model A (CNN+GRU)**: 
  - 配置文件：`configs/models/1_cnn_gru.yaml`
  - 编码器：`models/encoders/cnn_encoder.py` → `ResNetEncoder`
  - 解码器：`models/decoders/rnn_decoder.py` → `GRUDecoder`
  - 组装：`models/__init__.py` 第45-61行

- **Model B (Attn-GRU)**: 
  - 配置文件：`configs/models/2_attn_gru.yaml`
  - 编码器：`models/encoders/cnn_encoder.py` → `ResNetEncoder`（与Model A相同）
  - 解码器：`models/decoders/rnn_decoder.py` → `AttnGRUDecoder`（带注意力）
  - 组装：`models/__init__.py` 第63-80行

- **Model C (Region-Transformer)**: 
  - 配置文件：`configs/models/3_region_trans.yaml`
  - 编码器：`models/encoders/region_encoder.py` → `FasterRCNNEncoder`
  - 解码器：`models/decoders/trans_decoder.py` → `TransformerDecoder`
  - 组装：`models/__init__.py` 第82-101行

- **Model D (ViT-Transformer)**: 
  - 配置文件：`configs/models/4_vit_trans.yaml`
  - 编码器：`models/encoders/vit_encoder.py` → `ViTEncoder`
  - 解码器：`models/decoders/trans_decoder.py` → `TransformerDecoder`（与Model C相同）
  - 组装：`models/__init__.py` 第103-123行

- **Model E (Graph-Transformer)**: 
  - 配置文件：`configs/models/5_graph_trans.yaml`
  - 编码器：`models/encoders/graph_encoder.py` → `GCNEncoder`
  - 解码器：`models/decoders/trans_decoder.py` → `TransformerDecoder`（与Model C, D相同）
  - 组装：`models/__init__.py` 第125-147行

#### 设计优势

1. **配置驱动**：无需修改代码，只需修改YAML配置即可切换模型
2. **模块化**：Encoder和Decoder完全独立，可自由组合
3. **统一接口**：所有模型都通过`ImageCaptioner`统一接口使用
4. **易于扩展**：添加新模型只需创建新的Encoder/Decoder类，在`create_model()`中添加分支，创建配置文件

## 🎨 模型架构

项目包含5个完整的模型架构，从基线到前沿：

### Model A: CNN+GRU
- **编码器**：ResNet-50（全局特征）
- **解码器**：单层GRU
- **配置文件**：`configs/models/1_cnn_gru.yaml`
- **特点**：快速训练，基础性能，适合快速原型
- **适用场景**：基线对比，快速验证

### Model B: Attn-GRU
- **编码器**：ResNet-50
- **解码器**：带空间注意力的GRU
- **配置文件**：`configs/models/2_attn_gru.yaml`
- **特点**：引入空间注意力机制，提升特征聚焦能力
- **适用场景**：验证注意力机制的有效性

### Model C: Region-Transformer（主力模型）
- **编码器**：Faster R-CNN（区域特征，36个区域）
- **解码器**：Transformer（6层编码器 + 6层解码器）
- **配置文件**：`configs/models/3_region_trans.yaml`
- **特点**：区域特征 + Transformer长文本建模，性能最佳
- **适用场景**：主力模型，推荐用于RL微调

### Model D: ViT-Transformer
- **编码器**：Vision Transformer（ViT-Base）
- **解码器**：Transformer
- **配置文件**：`configs/models/4_vit_trans.yaml`
- **特点**：纯Transformer架构，端到端训练
- **适用场景**：验证纯Transformer架构的有效性

### Model E: Graph-Transformer
- **编码器**：图卷积网络（GCN，3层）
- **解码器**：Transformer
- **配置文件**：`configs/models/5_graph_trans.yaml`
- **特点**：处理拓扑结构信息，适合复杂搭配场景
- **适用场景**：验证图结构信息对描述的增益

### 模型对比总结

| 模型 | 编码器 | 解码器 | 训练速度 | 内存需求 | 生成质量 | 推荐度 |
|------|--------|--------|----------|----------|----------|--------|
| Model A | ResNet | GRU | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| Model B | ResNet | Attn-GRU | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Model C | FasterRCNN | Transformer | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Model D | ViT | Transformer | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Model E | GCN | Transformer | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

## 📚 依赖说明

主要依赖：
- `torch >= 2.1.0`
- `torchvision >= 0.16.0`
- `transformers >= 4.30.0`
- `numpy >= 1.24.0`
- `Pillow >= 10.0.0`
- `pyyaml >= 6.0`

完整依赖列表见 `requirements.txt`

## 🌐 启动前端界面

### 快速启动

#### 方式一：通过FastAPI服务器（推荐）

1. **启动后端服务器**
   ```bash
   # 在项目根目录下
   python api/server.py
   ```
   或使用uvicorn（支持热重载）：
   ```bash
   uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
   ```

2. **访问前端**
   服务器启动后，在浏览器中访问：
   ```
   http://localhost:8000/static/index.html
   ```

#### 方式二：直接打开HTML文件（仅用于开发测试）

如果只是测试前端界面（不调用API），可以直接打开HTML文件：

```bash
# Windows
start web/index.html

# Mac
open web/index.html

# Linux
xdg-open web/index.html
```

**注意**：直接打开HTML文件时，API调用会失败（CORS问题），只能查看界面效果。

### 启动前检查清单

1. **安装依赖**
   ```bash
   pip install fastapi uvicorn python-multipart
   ```
   或安装完整依赖：
   ```bash
   pip install -r requirements.txt
   ```

2. **检查模型文件**
   - 模型已训练完成（可选，如果未训练，API会返回错误但前端仍可访问）
   - 检查点文件路径正确：`checkpoints/{model_name}/best_model.pth`

3. **检查端口**
   默认端口是8000，如果被占用，可以修改 `api/server.py` 最后一行。

### 前端功能特性

- 🎨 **Cyberpunk风格**：深色主题，毛玻璃效果，蓝紫渐变文字
- 📤 **拖拽上传**：支持点击或拖拽上传图像
- ⚙️ **模型切换**：支持5个模型（CNN+GRU, Attn-GRU, Region-Trans, ViT-Trans, Graph-Trans）
- 🎯 **解码策略**：Greedy / Beam Search / Sampling
- ⌨️ **打字机效果**：描述逐字显示
- 🎨 **关键词高亮**：自动识别并高亮属性词（如 "red", "sleeveless", "denim"）
- 📊 **实时指标**：显示CIDEr分数和动态进度条
- 🔍 **X-Ray Vision**：点击描述中的单词查看注意力热力图

### 使用流程

1. **启动后端服务器**
   ```bash
   python api/server.py
   ```

2. **打开浏览器**
   访问 `http://localhost:8000/static/index.html`

3. **上传图像**
   - 点击或拖拽图像到上传区域
   - 选择模型（5个模型可选）
   - 选择解码策略（Greedy / Beam / Sampling）

4. **生成描述**
   - 点击"Generate Caption"按钮
   - 等待生成完成
   - 查看生成的描述和实时指标

5. **查看注意力**
   - 在生成的描述中点击任意单词
   - 查看X-Ray Vision模式的注意力热力图

### 配置说明

#### API地址配置

前端JavaScript中的API地址在 `web/js/main.js` 中配置：

```javascript
const API_BASE = 'http://localhost:8000/api';
```

如果后端运行在不同地址或端口，需要修改此配置。

#### 静态文件路径

FastAPI服务器会自动挂载 `web/` 目录为静态文件服务：

```python
# api/server.py
web_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'web')
if os.path.exists(web_dir):
    app.mount("/static", StaticFiles(directory=web_dir), name="static")
```

### 前端常见问题

1. **无法访问前端页面**
   - 检查后端服务器是否启动
   - 检查 `web/` 目录是否存在
   - 检查 `api/server.py` 中的静态文件挂载配置

2. **API调用失败**
   - 检查后端服务器日志，查看具体错误
   - 确认模型文件是否存在
   - 检查API地址配置是否正确（`web/js/main.js`）

3. **CORS错误**
   - FastAPI已配置CORS中间件，应该不会有此问题
   - 如果仍有问题，检查 `api/server.py` 中的CORS配置

4. **模型加载失败**
   - 确认模型已训练完成
   - 检查检查点文件路径
   - 如果未训练模型，可以先训练或使用预训练模型

## 🔍 故障排除

### 常见问题

1. **模型加载失败**
   - 检查配置文件路径是否正确
   - 确认词汇表大小匹配

2. **CUDA内存不足**
   - 减小batch_size
   - 使用CPU训练：`--device cpu`

3. **配置文件错误**
   - 检查YAML语法
   - 确认`_base_`路径正确

4. **前端无法访问**
   - 确认后端服务器已启动
   - 检查端口8000是否被占用
   - 确认`web/`目录存在

## 📄 许可证

本项目仅供学习和研究使用。

## 🙏 致谢

- DeepFashion-MultiModal数据集
- PyTorch社区
- 相关论文作者

---

**注意**：本项目已按照设计文档完成重构，采用配置驱动、模块化的架构。所有模型通过统一的接口和训练流程进行管理。

