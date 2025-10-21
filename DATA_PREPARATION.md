# 数据准备指南

## 1. 数据集下载

### DeepFashion-MultiModal数据集

**官方信息：**
- 论文：https://arxiv.org/abs/1807.10998
- GitHub：https://github.com/switchablenorms/DeepFashion-MultiModal
- 数据集大小：约2GB

**下载步骤：**
1. 访问官方GitHub页面
2. 按照README中的说明申请数据集
3. 下载数据集文件（通常是ZIP格式）

## 2. 数据目录结构

下载完成后，需要将数据整理成以下结构：

```
data/
└── DeepFashion-MultiModal/
    ├── images/                    # 原始图像文件
    │   ├── 000001.jpg
    │   ├── 000002.jpg
    │   └── ...
    ├── captions/                  # 描述文件（JSON格式）
    │   ├── 000001.json
    │   ├── 000002.json
    │   └── ...
    ├── regions/                   # 区域特征文件（需要预提取）
    │   ├── 000001.npy
    │   ├── 000002.npy
    │   └── ...
    ├── train_list.txt            # 训练集文件列表
    ├── val_list.txt              # 验证集文件列表
    └── test_list.txt             # 测试集文件列表
```

## 3. 数据预处理步骤

### 步骤1：解压和组织数据
```bash
# 解压下载的数据集
unzip DeepFashion-MultiModal.zip

# 创建标准目录结构
mkdir -p data/DeepFashion-MultiModal/{images,captions,regions}
```

### 步骤2：创建数据分割
运行数据预处理脚本：
```bash
python scripts/download_and_prepare_data.py --data_dir data/DeepFashion-MultiModal --prepare
```

### 步骤3：提取区域特征（用于Transformer模型）
```python
# 使用预训练的Faster R-CNN提取区域特征
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import numpy as np
from PIL import Image
import os

def extract_region_features(image_dir, output_dir, max_regions=36):
    """提取区域特征"""
    # 加载预训练模型
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    
    # 图像预处理
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    
    for image_file in os.listdir(image_dir):
        if image_file.endswith(('.jpg', '.jpeg', '.png')):
            # 加载图像
            image_path = os.path.join(image_dir, image_file)
            image = Image.open(image_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0)
            
            # 提取特征
            with torch.no_grad():
                features = model.backbone(image_tensor)
                # 这里需要进一步处理特征...
            
            # 保存特征
            feature_file = os.path.splitext(image_file)[0] + '.npy'
            np.save(os.path.join(output_dir, feature_file), features.numpy())
```

## 4. 描述文件格式

每个描述文件应该是JSON格式，包含多个描述：

```json
{
    "captions": [
        "A woman wearing a red dress",
        "Fashionable red dress for summer",
        "Elegant red evening dress"
    ]
}
```

## 5. 验证数据完整性

运行以下脚本验证数据：
```bash
python scripts/verify_data.py --data_dir data/DeepFashion-MultiModal
```

## 6. 常见问题

### Q: 如何获取数据集？
A: 访问官方GitHub页面，按照说明申请数据集。通常需要填写申请表格。

### Q: 数据集太大，如何管理？
A: 可以使用数据加载器按需加载，或者使用数据压缩。

### Q: 区域特征提取需要多长时间？
A: 取决于硬件配置，通常需要几小时到一天时间。

### Q: 如何处理内存不足？
A: 可以减少批次大小，或者使用数据流式加载。

## 7. 替代数据集

如果无法获取DeepFashion-MultiModal数据集，可以考虑以下替代方案：

1. **Fashion-MNIST**: 简单的服装分类数据集
2. **DeepFashion**: 原始DeepFashion数据集
3. **FashionVC**: 服装视觉问答数据集
4. **自定义数据集**: 收集自己的服装图像和描述

## 8. 数据增强

为了提高模型性能，可以使用以下数据增强技术：

```python
# 图像增强
transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((256, 256)),
    torchvision.transforms.RandomCrop((224, 224)),
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
])
```

## 9. 下一步

数据准备完成后，可以：
1. 运行数据分析笔记本：`notebooks/data_analysis.ipynb`
2. 开始训练模型：`python scripts/train_cnn_gru.py`
3. 进行模型评测：`python scripts/evaluate.py`
