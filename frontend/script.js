// DOM元素
const uploadBox = document.getElementById('uploadBox');
const imageInput = document.getElementById('imageInput');
const previewImage = document.getElementById('previewImage');
const generateBtn = document.getElementById('generateBtn');
const clearBtn = document.getElementById('clearBtn');
const resultSection = document.getElementById('resultSection');
const captionBox = document.getElementById('captionBox');
const modelSelect = document.getElementById('modelSelect');
const copyBtn = document.getElementById('copyBtn');

let selectedImage = null;

// 点击上传框触发文件选择
uploadBox.addEventListener('click', () => {
    imageInput.click();
});

// 文件选择事件
imageInput.addEventListener('change', (e) => {
    handleFile(e.target.files[0]);
});

// 拖拽上传
uploadBox.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadBox.classList.add('dragover');
});

uploadBox.addEventListener('dragleave', () => {
    uploadBox.classList.remove('dragover');
});

uploadBox.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadBox.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        handleFile(file);
    } else {
        alert('请上传图片文件！');
    }
});

// 处理选择的文件
function handleFile(file) {
    if (!file) return;
    
    // 验证文件类型
    if (!file.type.startsWith('image/')) {
        alert('请选择图片文件！');
        return;
    }
    
    // 验证文件大小（10MB）
    if (file.size > 10 * 1024 * 1024) {
        alert('文件大小不能超过10MB！');
        return;
    }
    
    selectedImage = file;
    
    // 显示预览
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        resultSection.style.display = 'grid';
        captionBox.innerHTML = '<p class="loading">请选择模型并点击"生成描述"按钮</p>';
        generateBtn.disabled = false;
        clearBtn.disabled = false;
        
        // 滚动到结果区域
        resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    };
    reader.readAsDataURL(file);
}

// 生成描述
generateBtn.addEventListener('click', async () => {
    if (!selectedImage) {
        alert('请先上传图片！');
        return;
    }
    
    const selectedModel = modelSelect.value;
    generateBtn.disabled = true;
    captionBox.innerHTML = '<p class="loading">正在生成描述，请稍候...</p>';
    
    try {
        // 创建FormData
        const formData = new FormData();
        formData.append('image', selectedImage);
        formData.append('model', selectedModel);
        
        // 发送请求到后端API
        const response = await fetch('http://localhost:5000/api/generate', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error('生成描述失败');
        }
        
        const data = await response.json();
        
        if (data.success) {
            captionBox.innerHTML = `<p>${data.caption}</p>`;
        } else {
            throw new Error(data.error || '生成描述失败');
        }
        
    } catch (error) {
        console.error('Error:', error);
        captionBox.innerHTML = `<p style="color: #dc3545;">生成失败: ${error.message}<br>请确保后端服务正在运行</p>`;
    } finally {
        generateBtn.disabled = false;
    }
});

// 清除
clearBtn.addEventListener('click', () => {
    selectedImage = null;
    imageInput.value = '';
    previewImage.src = '';
    resultSection.style.display = 'none';
    captionBox.innerHTML = '';
    generateBtn.disabled = true;
    clearBtn.disabled = true;
    uploadBox.classList.remove('dragover');
});

// 复制描述
copyBtn.addEventListener('click', () => {
    const captionText = captionBox.querySelector('p')?.textContent;
    if (!captionText || captionText.includes('正在生成') || captionText.includes('请选择')) {
        alert('没有可复制的描述！');
        return;
    }
    
    navigator.clipboard.writeText(captionText).then(() => {
        // 显示成功提示
        const successMsg = document.createElement('div');
        successMsg.className = 'success-message show';
        successMsg.textContent = '✅ 描述已复制到剪贴板！';
        captionBox.appendChild(successMsg);
        
        setTimeout(() => {
            successMsg.remove();
        }, 2000);
    }).catch(err => {
        console.error('复制失败:', err);
        alert('复制失败，请手动复制');
    });
});

// 页面加载完成后的初始化
document.addEventListener('DOMContentLoaded', () => {
    console.log('服饰图像描述生成系统已加载');
    console.log('请确保后端API服务正在运行在 http://localhost:5000');
});
