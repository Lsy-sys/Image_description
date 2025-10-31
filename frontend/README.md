# 前端使用说明

## 📁 文件结构

```
frontend/
├── index.html      # 主页面
├── styles.css      # 样式文件
├── script.js       # JavaScript功能
└── README.md       # 说明文档
```

## 🚀 使用方法

### 方法1：直接打开HTML文件

1. 双击 `index.html` 文件在浏览器中打开
2. 上传图像并生成描述
3. **注意**：需要启动后端API服务（见下方说明）

### 方法2：使用本地服务器（推荐）

#### Python HTTP服务器
```bash
# 在frontend目录下
cd frontend
python -m http.server 8000
```

然后在浏览器中访问：`http://localhost:8000`

#### Node.js HTTP服务器
```bash
# 安装http-server（如果未安装）
npm install -g http-server

# 在frontend目录下启动
cd frontend
http-server -p 8000
```

## 🔌 后端API服务

前端需要通过API与后端Python服务通信。请确保启动后端服务：

```bash
# 在项目根目录下
python frontend/api_server.py
```

后端服务默认运行在 `http://localhost:5000`

## 📝 功能说明

### 主要功能
- ✅ 图像上传（点击或拖拽）
- ✅ 图像预览
- ✅ 模型选择（CNN+GRU / Transformer）
- ✅ 描述生成
- ✅ 描述复制
- ✅ 响应式设计

### 支持的图像格式
- JPEG (.jpg, .jpeg)
- PNG (.png)
- 最大文件大小：10MB

### 浏览器兼容性
- Chrome / Edge (推荐)
- Firefox
- Safari

## 🎨 界面特点

- **现代化设计**：渐变背景、圆角卡片、平滑动画
- **响应式布局**：适配桌面和移动设备
- **用户友好**：直观的操作界面和实时反馈
- **美观大方**：专业的配色和排版

## ⚙️ 配置说明

如果需要修改API地址，请编辑 `script.js` 文件：

```javascript
// 修改这一行的URL
const response = await fetch('http://localhost:5000/api/generate', {
    method: 'POST',
    body: formData
});
```

## 🐛 故障排除

### 问题1：无法生成描述
- 检查后端API服务是否运行
- 检查浏览器控制台是否有错误信息
- 确认API地址是否正确

### 问题2：图像上传失败
- 检查图像格式是否支持
- 检查文件大小是否超过10MB
- 检查浏览器是否支持File API

### 问题3：样式显示异常
- 清除浏览器缓存
- 检查CSS文件是否正确加载
- 使用现代浏览器（Chrome/Firefox/Edge）

## 📚 技术栈

- **HTML5**：页面结构
- **CSS3**：样式和动画
- **JavaScript (ES6+)**：交互功能
- **Fetch API**：与后端通信
- **File API**：文件处理
