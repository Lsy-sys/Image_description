#!/usr/bin/env python3
"""
前端API服务器
提供图像描述生成的HTTP API接口
"""

import os
import sys
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
from PIL import Image
import io

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cnn_gru.model import CNNGruModel
from models.transformer.model import TransformerModel
from data.transforms import ImageTransforms

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)  # 允许跨域请求

# 全局变量
cnn_gru_model = None
transformer_model = None
cnn_gru_vocab = None
transformer_vocab = None
cnn_gru_extractor = None
transformer_extractor = None
device = None

# 图像变换
transform = ImageTransforms(224, is_training=False)

def load_models():
    """加载模型"""
    global cnn_gru_model, transformer_model
    global cnn_gru_vocab, transformer_vocab
    global device, cnn_gru_extractor, transformer_extractor
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载CNN+GRU模型
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cnn_gru_path = os.path.join(project_root, 'checkpoints/cnn_gru/best_model.pth')
    if os.path.exists(cnn_gru_path):
        print(f"加载CNN+GRU模型: {cnn_gru_path}")
        checkpoint = torch.load(cnn_gru_path, map_location=device)
        cnn_gru_vocab = checkpoint['vocab']
        
        cnn_gru_model = CNNGruModel(
            embed_size=512,
            hidden_size=512,
            vocab_size=len(cnn_gru_vocab),
            num_layers=1,
            pretrained=True
        ).to(device)
        
        cnn_gru_model.load_state_dict(checkpoint['model_state_dict'])
        cnn_gru_model.eval()
        print("CNN+GRU模型加载完成")
    else:
        print(f"警告: CNN+GRU模型文件不存在: {cnn_gru_path}")
    
    # 加载Transformer模型
    transformer_path = os.path.join(project_root, 'checkpoints/transformer/best_model.pth')
    if os.path.exists(transformer_path):
        print(f"加载Transformer模型: {transformer_path}")
        checkpoint = torch.load(transformer_path, map_location=device)
        transformer_vocab = checkpoint['vocab']
        
        transformer_model = TransformerModel(
            vocab_size=len(transformer_vocab),
            d_model=512,
            num_heads=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            d_ff=2048,
            dropout=0.1,
            max_len=100
        ).to(device)
        
        transformer_model.load_state_dict(checkpoint['model_state_dict'])
        transformer_model.eval()
        
        # 创建区域特征提取器
        import torchvision.models as models
        import torch.nn as nn
        backbone = models.resnet50(pretrained=True)
        backbone = nn.Sequential(*list(backbone.children())[:-1])
        backbone.eval()
        backbone.to(device)
        transformer_extractor = backbone
        
        print("Transformer模型加载完成")
    else:
        print(f"警告: Transformer模型文件不存在: {transformer_path}")

def generate_caption_cnn_gru(image_tensor):
    """使用CNN+GRU模型生成描述"""
    global cnn_gru_model, cnn_gru_vocab
    
    if cnn_gru_model is None or cnn_gru_vocab is None:
        return None
    
    with torch.no_grad():
        features = cnn_gru_model.encoder(image_tensor)
        generated = cnn_gru_model.decoder.sample(features, max_length=50, vocab=cnn_gru_vocab)
        
        caption = []
        for word_id in generated[0]:
            word = cnn_gru_vocab.idx2word[word_id.item()]
            if word_id.item() == cnn_gru_vocab.eos_idx:
                break
            if word not in [cnn_gru_vocab.SOS_TOKEN, cnn_gru_vocab.PAD_TOKEN, cnn_gru_vocab.UNK_TOKEN]:
                caption.append(word)
        
        return ' '.join(caption)

def generate_caption_transformer(image_tensor):
    """使用Transformer模型生成描述"""
    global transformer_model, transformer_vocab, transformer_extractor
    
    if transformer_model is None or transformer_vocab is None or transformer_extractor is None:
        return None
    
    with torch.no_grad():
        # 提取区域特征
        global_features = transformer_extractor(image_tensor)
        # 处理不同的特征形状
        if len(global_features.shape) == 4:
            global_features = global_features.squeeze(-1).squeeze(-1)
        elif len(global_features.shape) == 2:
            pass  # 已经是2D了
        else:
            global_features = global_features.view(global_features.size(0), -1)
        
        region_features = global_features.unsqueeze(1).repeat(1, 36, 1)
        noise = torch.randn_like(region_features) * 0.1
        region_features = region_features + noise
        
        # 生成描述
        generated = transformer_model.generate(region_features, transformer_vocab, max_length=50, temperature=1.0)
        
        caption = []
        for word_id in generated[0]:
            word = transformer_vocab.idx2word[word_id.item()]
            if word_id.item() == transformer_vocab.eos_idx:
                break
            if word not in [transformer_vocab.SOS_TOKEN, transformer_vocab.PAD_TOKEN, transformer_vocab.UNK_TOKEN]:
                caption.append(word)
        
        return ' '.join(caption)

@app.route('/')
def index():
    """返回前端页面"""
    return send_from_directory('.', 'index.html')

@app.route('/api/generate', methods=['POST'])
def generate():
    """生成图像描述的API接口"""
    try:
        # 检查是否有文件
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': '未上传图像'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': '未选择图像'}), 400
        
        # 获取模型类型
        model_type = request.form.get('model', 'cnn_gru')
        
        # 加载并预处理图像
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        image_tensor = transform.get_transforms()(image).unsqueeze(0).to(device)
        
        # 生成描述
        if model_type == 'cnn_gru':
            caption = generate_caption_cnn_gru(image_tensor)
            if caption is None:
                return jsonify({'success': False, 'error': 'CNN+GRU模型未加载'}), 500
        elif model_type == 'transformer':
            caption = generate_caption_transformer(image_tensor)
            if caption is None:
                return jsonify({'success': False, 'error': 'Transformer模型未加载'}), 500
        else:
            return jsonify({'success': False, 'error': '未知的模型类型'}), 400
        
        if not caption:
            caption = "未能生成描述，请尝试其他图像或模型"
        
        return jsonify({
            'success': True,
            'caption': caption,
            'model': model_type
        })
        
    except Exception as e:
        print(f"生成描述时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'生成失败: {str(e)}'
        }), 500

@app.route('/api/status', methods=['GET'])
def status():
    """获取服务状态"""
    return jsonify({
        'cnn_gru_loaded': cnn_gru_model is not None,
        'transformer_loaded': transformer_model is not None,
        'device': str(device)
    })

if __name__ == '__main__':
    print("=" * 50)
    print("启动前端API服务器...")
    print("=" * 50)
    
    # 加载模型
    load_models()
    
    print("\n" + "=" * 50)
    print("服务器启动完成！")
    print("前端地址: http://localhost:5000")
    print("API地址: http://localhost:5000/api/generate")
    print("=" * 50)
    print("\n按 Ctrl+C 停止服务器\n")
    
    # 启动Flask服务器
    app.run(host='0.0.0.0', port=5000, debug=True)
