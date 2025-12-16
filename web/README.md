# 前端使用说明

## 🎨 前端界面

前端采用Cyberpunk风格设计，支持所有5个模型的切换和使用。

## 📋 支持的模型

前端支持以下5个模型：

1. **CNN+GRU** (Model A) - 基线模型
2. **Attn-GRU** (Model B) - 带空间注意力的GRU
3. **Region-Trans** (Model C) - 区域特征+Transformer（主力模型）
4. **ViT-Trans** (Model D) - Vision Transformer
5. **Graph-Trans** (Model E) - 图卷积+Transformer

## 🚀 启动方式

### 1. 启动后端服务器

```bash
python api/server.py
```

### 2. 访问前端

在浏览器中打开：
```
http://localhost:8000/static/index.html
```

## 🎯 功能说明

### 模型切换器

- 点击任意模型按钮即可切换
- 当前选中的模型会高亮显示
- 所有5个模型都可以选择

### 解码策略

- **Greedy**: 贪婪解码（最快，但可能重复）
- **Beam Search**: 束搜索（平衡质量和速度）
- **Sampling**: 随机采样（最多样，但可能不准确）

### 交互功能

1. **图像上传**：拖拽或点击上传
2. **生成描述**：点击"Generate Caption"按钮
3. **打字机效果**：描述逐字显示
4. **关键词高亮**：自动识别并高亮属性词
5. **注意力可视化**：点击描述中的单词查看注意力热力图

## 📁 文件结构

```
web/
├── index.html      # 主页面
├── styles.css      # Cyberpunk风格样式
├── js/
│   └── main.js     # 前端交互逻辑
└── assets/         # 静态资源（可选）
```

## 🔧 自定义配置

### 修改API地址

编辑 `web/js/main.js`：

```javascript
const API_BASE = 'http://localhost:8000/api';  // 修改为你的API地址
```

### 修改模型按钮

编辑 `web/index.html`，在Model Switcher部分：

```html
<button class="model-btn" data-model="cnn_gru">CNN+GRU</button>
```

`data-model` 属性必须与后端API支持的模型类型一致。

## 🐛 常见问题

### 模型按钮显示不全

- 检查浏览器窗口宽度
- 在小屏幕上，按钮会以3列网格显示
- 在大屏幕上，按钮会以5列网格显示

### 模型切换无效

- 检查浏览器控制台是否有错误
- 确认后端API支持该模型类型
- 检查 `data-model` 属性是否正确

## 📝 注意事项

- 确保后端服务器已启动
- 模型需要先训练才能使用（否则API会返回错误）
- 前端界面可以正常访问，但生成功能需要训练好的模型

