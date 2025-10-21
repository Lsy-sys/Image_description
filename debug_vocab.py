#!/usr/bin/env python3
"""
调试词汇表和生成过程
"""

import os
import sys
import torch
from PIL import Image

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.cnn_gru.model import CNNGruModel
from data.transforms import ImageTransforms

def debug_vocab_and_generation():
    """调试词汇表和生成过程"""
    
    # 加载模型
    model_path = "checkpoints/cnn_gru/best_model.pth"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(model_path, map_location=device)
    vocab = checkpoint['vocab']
    
    print(f"词汇表大小: {len(vocab)}")
    print(f"特殊标记:")
    print(f"  PAD: {vocab.PAD_TOKEN} (idx: {vocab.word2idx[vocab.PAD_TOKEN]})")
    print(f"  UNK: {vocab.UNK_TOKEN} (idx: {vocab.word2idx[vocab.UNK_TOKEN]})")
    print(f"  SOS: {vocab.SOS_TOKEN} (idx: {vocab.word2idx[vocab.SOS_TOKEN]})")
    print(f"  EOS: {vocab.EOS_TOKEN} (idx: {vocab.word2idx[vocab.EOS_TOKEN]})")
    
    print(f"\n前20个词汇:")
    for i in range(min(20, len(vocab))):
        word = vocab.idx2word[i]
        print(f"  {i}: {word}")
    
    print(f"\n最后10个词汇:")
    for i in range(max(0, len(vocab)-10), len(vocab)):
        word = vocab.idx2word[i]
        print(f"  {i}: {word}")
    
    # 创建模型
    model = CNNGruModel(
        embed_size=512,
        hidden_size=512,
        vocab_size=len(vocab),
        num_layers=1,
        pretrained=True
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 测试图像
    image_path = "data/DeepFashion-MultiModal/images/MEN-Denim-id_00000080-01_7_additional.jpg"
    transform = ImageTransforms(224, is_training=False)
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform.get_transforms()(image).unsqueeze(0).to(device)
    
    print(f"\n开始生成描述...")
    
    with torch.no_grad():
        # 提取图像特征
        features = model.encoder(image_tensor)
        print(f"图像特征形状: {features.shape}")
        
        # 详细生成过程
        print(f"\n逐步生成过程:")
        batch_size = 1
        device = features.device
        
        generated = []
        input_word = torch.tensor([vocab.sos_idx] * batch_size, device=device)
        hidden = None
        
        for step in range(30):  # 增加最大长度
            # 词嵌入
            embedded = model.decoder.embed(input_word.unsqueeze(1))
            
            # 结合图像特征
            image_features = features.unsqueeze(1)
            gru_input = embedded + image_features
            
            # GRU前向传播
            output, hidden = model.decoder.gru(gru_input, hidden)
            
            # 输出层
            output = model.decoder.linear(output.squeeze(1))
            
            # 获取概率分布
            probs = torch.softmax(output, dim=1)
            top_probs, top_indices = torch.topk(probs, 5, dim=1)
            
            # 选择最可能的词
            predicted = output.argmax(1)
            word = vocab.idx2word[predicted.item()]
            
            print(f"步骤 {step+1}: {word} (idx: {predicted.item()}, prob: {probs[0, predicted.item()]:.4f})")
            print(f"  前5个候选: {[vocab.idx2word[idx.item()] for idx in top_indices[0]]}")
            print(f"  对应概率: {[f'{prob:.4f}' for prob in top_probs[0]]}")
            
            generated.append(predicted)
            
            # 更新输入
            input_word = predicted
            
            # 如果生成了结束标记，停止
            if predicted.item() == vocab.eos_idx:
                print(f"  遇到结束标记，停止生成")
                break
        
        # 转换为文本
        caption = []
        for word_id in generated:
            word = vocab.idx2word[word_id.item()]
            if word == vocab.EOS_TOKEN:
                break
            if word not in [vocab.SOS_TOKEN, vocab.PAD_TOKEN, vocab.UNK_TOKEN]:
                caption.append(word)
        
        result = ' '.join(caption)
        print(f"\n最终生成的描述: {result}")

if __name__ == '__main__':
    debug_vocab_and_generation()
