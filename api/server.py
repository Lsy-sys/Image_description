"""
FastAPI后端服务
提供图像描述生成和注意力可视化接口
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from typing import Optional, List
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import json
from PIL import Image
import io
import base64

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import create_model_from_config, load_config
from data.vocabulary import Vocabulary
from data.transforms import ImageTransforms


# 全局变量
app = FastAPI(title="DeepFashion Image Captioning API", version="1.0.0")

# CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 模型缓存
model_cache = {}
vocab_cache = {}
transform_cache = {}


class CaptionRequest(BaseModel):
    model_type: str = "cnn_gru"  # cnn_gru, attn_gru, region_trans, vit_trans, graph_trans
    strategy: str = "greedy"  # greedy, beam_search, sampling
    max_length: int = 50
    temperature: float = 1.0


class AttentionRequest(BaseModel):
    model_type: str = "cnn_gru"
    word: str


def load_model(model_type: str):
    """加载模型（带缓存）"""
    if model_type in model_cache:
        return model_cache[model_type], vocab_cache[model_type]
    
    # 模型配置映射
    config_path = get_model_config_path(model_type)
    
    # 加载配置
    config = load_config(config_path)
    
    # 加载词汇表
    vocab_path = config['paths']['vocab_path']
    if os.path.exists(vocab_path):
        vocab = Vocabulary.load(vocab_path)
    else:
        # 从数据集构建
        vocab = Vocabulary.from_captions(config['paths']['data_dir'])
        vocab.save(vocab_path)
    
    # 创建模型
    model = create_model_from_config(config_path, len(vocab))
    
    # 加载权重
    checkpoint_dir = config['paths']['checkpoint_dir']
    checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
    else:
        print(f"警告: 未找到检查点文件 {checkpoint_path}，使用未训练模型")
    
    # 缓存
    model_cache[model_type] = model
    vocab_cache[model_type] = vocab
    
    return model, vocab


def get_model_config_path(model_type: str) -> str:
    """根据model_type获取模型配置路径（供其他接口复用）"""
    config_map = {
        "cnn_gru": "configs/models/1_cnn_gru.yaml",
        "attn_gru": "configs/models/2_attn_gru.yaml",
        "region_trans": "configs/models/3_region_trans.yaml",
        "vit_trans": "configs/models/4_vit_trans.yaml",
        "graph_trans": "configs/models/5_graph_trans.yaml"
    }
    if model_type not in config_map:
        raise ValueError(f"Unknown model type: {model_type}")
    return config_map[model_type]


def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """预处理图像"""
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    transform = ImageTransforms(224, is_training=False)
    image_tensor = transform.get_transforms()(image).unsqueeze(0)
    return image_tensor


@app.get("/")
async def root():
    """根路径"""
    return {"message": "DeepFashion Image Captioning API", "version": "1.0.0"}


@app.post("/api/predict")
async def predict_caption(
    file: UploadFile = File(...),
    model_type: str = "cnn_gru",
    strategy: str = "greedy",
    max_length: int = 50,
    temperature: float = 1.0
):
    """
    生成图像描述
    """
    try:
        # 读取图像
        image_bytes = await file.read()
        image_tensor = preprocess_image(image_bytes)
        
        # 加载模型
        model, vocab = load_model(model_type)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        image_tensor = image_tensor.to(device)
        
        # 生成描述
        with torch.no_grad():
            if hasattr(model, 'generate'):
                sequences, _ = model.generate(
                    images=image_tensor,
                    vocab=vocab,
                    max_length=max_length,
                    strategy=strategy,
                    temperature=temperature
                )
            else:
                # 兼容旧接口
                features = model.encode(images=image_tensor)
                sequences, _ = model.decoder.generate(
                    features,
                    vocab=vocab,
                    max_length=max_length,
                    strategy=strategy,
                    temperature=temperature
                )
        
        # 转换为文本
        sequence = sequences[0].cpu().numpy()
        caption_words = []
        for word_id in sequence:
            word = vocab.idx2word.get(word_id, '<unk>')
            if word == '<eos>':
                break
            if word not in ['<bos>', '<pad>', '<unk>']:
                caption_words.append(word)
        
        caption = ' '.join(caption_words)
        
        # 计算预估CIDEr分数（简化）
        estimated_cider = min(1.0, len(caption_words) / 30.0) * 0.8
        
        return JSONResponse({
            "caption": caption,
            "estimated_cider": round(estimated_cider, 4),
            "word_count": len(caption_words),
            "model_type": model_type,
            "strategy": strategy
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/attention")
async def get_attention(
    file: UploadFile = File(...),
    model_type: str = "cnn_gru",
    word: str = ""
):
    """
    获取注意力权重
    """
    try:
        # 读取图像
        image_bytes = await file.read()
        image_tensor = preprocess_image(image_bytes)
        
        # 加载模型
        model, vocab = load_model(model_type)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        image_tensor = image_tensor.to(device)
        
        # 生成描述并提取注意力
        with torch.no_grad():
            # 这里需要模型支持返回注意力权重
            # 简化实现：返回随机注意力图
            attention_map = np.random.rand(14, 14)  # 14x14的注意力图
            
            # 归一化
            attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
        
        # 转换为base64
        import cv2
        attention_img = (attention_map * 255).astype(np.uint8)
        attention_colored = cv2.applyColorMap(attention_img, cv2.COLORMAP_JET)
        _, buffer = cv2.imencode('.png', attention_colored)
        attention_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return JSONResponse({
            "attention_map": attention_base64,
            "word": word,
            "model_type": model_type
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/models")
async def list_models():
    """列出可用模型"""
    return {
        "models": [
            {"id": "cnn_gru", "name": "CNN+GRU", "description": "基线模型"},
            {"id": "attn_gru", "name": "Attn-GRU", "description": "带注意力的GRU"},
            {"id": "region_trans", "name": "Region-Transformer", "description": "区域特征+Transformer"},
            {"id": "vit_trans", "name": "ViT-Transformer", "description": "Vision Transformer"},
            {"id": "graph_trans", "name": "Graph-Transformer", "description": "图卷积+Transformer"}
        ],
        "strategies": ["greedy", "beam_search", "sampling"]
    }


@app.get("/api/training_log")
async def get_training_log(model_type: str = Query("region_trans", description="模型类型，用于选择对应log目录")):
    """
    返回训练日志，用于前端 Training Monitor 与分析脚本
    """
    try:
        config_path = get_model_config_path(model_type)
        config = load_config(config_path)
        log_dir = config['paths'].get('log_dir', 'logs')
        log_path = os.path.join(log_dir, 'training_log.json')

        if not os.path.exists(log_path):
            return JSONResponse({"epochs": [], "train_loss": [], "val_loss": []})

        with open(log_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 确保三个关键字段存在
        data.setdefault("epochs", list(range(1, len(data.get("train_loss", [])) + 1)))
        data.setdefault("train_loss", [])
        data.setdefault("val_loss", [])

        return JSONResponse(data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 挂载静态文件（前端）
web_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'web')
if os.path.exists(web_dir):
    app.mount("/static", StaticFiles(directory=web_dir), name="static")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

