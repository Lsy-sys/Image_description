"""
词汇表构建和管理
"""

import json
import pickle
from collections import Counter
import numpy as np


class Vocabulary:
    """词汇表类"""
    
    def __init__(self, min_freq=5):
        """
        Args:
            min_freq: 最小词频阈值
        """
        self.min_freq = min_freq
        self.word2idx = {}
        self.idx2word = {}
        self.word_freq = Counter()
        
        # 特殊标记
        self.PAD_TOKEN = '<pad>'
        self.UNK_TOKEN = '<unk>'
        self.SOS_TOKEN = '<sos>'
        self.EOS_TOKEN = '<eos>'
        
        # 添加特殊标记
        self._add_special_tokens()
    
    def _add_special_tokens(self):
        """添加特殊标记"""
        special_tokens = [
            self.PAD_TOKEN,  # 0
            self.UNK_TOKEN,  # 1
            self.SOS_TOKEN,  # 2
            self.EOS_TOKEN   # 3
        ]
        
        for i, token in enumerate(special_tokens):
            self.word2idx[token] = i
            self.idx2word[i] = token
    
    def __len__(self):
        """返回词汇表大小"""
        return len(self.word2idx)
    
    def build_vocab(self, captions_list):
        """
        构建词汇表
        Args:
            captions_list: 所有描述文本列表
        """
        print("构建词汇表...")
        
        # 统计词频
        for captions in captions_list:
            for caption in captions:
                words = self._tokenize(caption)
                self.word_freq.update(words)
        
        # 添加高频词
        for word, freq in self.word_freq.items():
            if freq >= self.min_freq:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
        
        print(f"词汇表大小: {len(self.word2idx)}")
        print(f"过滤掉的低频词数量: {len(self.word_freq) - len(self.word2idx) + 4}")
    
    def build_vocab_from_dataset(self, data_dir):
        """从数据集构建词汇表"""
        import os
        import json
        
        print("从数据集构建词汇表...")
        captions_list = []
        
        # 遍历所有分割
        for split in ['train', 'val', 'test']:
            list_file = os.path.join(data_dir, f'{split}_list.txt')
            if not os.path.exists(list_file):
                continue
                
            with open(list_file, 'r', encoding='utf-8') as f:
                data_list = [line.strip() for line in f.readlines()]
            
            for item_id in data_list:
                text_path = os.path.join(data_dir, 'captions', f'{item_id}.json')
                if os.path.exists(text_path):
                    with open(text_path, 'r', encoding='utf-8') as f:
                        caption_data = json.load(f)
                    captions = caption_data.get('captions', [])
                    if captions:
                        captions_list.append(captions)
        
        # 构建词汇表
        self.build_vocab(captions_list)
    
    def _tokenize(self, text):
        """简单的分词"""
        # 这里可以使用更复杂的分词方法
        return text.lower().split()
    
    def encode(self, text, max_length=None):
        """
        将文本编码为索引序列
        Args:
            text: 输入文本
            max_length: 最大长度
        Returns:
            索引序列
        """
        words = self._tokenize(text)
        indices = []
        
        # 添加开始标记
        indices.append(self.word2idx[self.SOS_TOKEN])
        
        # 添加词索引
        for word in words:
            if word in self.word2idx:
                indices.append(self.word2idx[word])
            else:
                indices.append(self.word2idx[self.UNK_TOKEN])
        
        # 添加结束标记
        indices.append(self.word2idx[self.EOS_TOKEN])
        
        # 截断或填充
        if max_length:
            if len(indices) > max_length:
                indices = indices[:max_length]
            else:
                # 填充
                while len(indices) < max_length:
                    indices.append(self.word2idx[self.PAD_TOKEN])
        
        return indices
    
    def decode(self, indices):
        """
        将索引序列解码为文本
        Args:
            indices: 索引序列
        Returns:
            文本字符串
        """
        words = []
        for idx in indices:
            if idx in self.idx2word:
                word = self.idx2word[idx]
                if word in [self.SOS_TOKEN, self.EOS_TOKEN, self.PAD_TOKEN]:
                    continue
                words.append(word)
            else:
                words.append(self.UNK_TOKEN)
        
        return ' '.join(words)
    
    def save(self, filepath):
        """保存词汇表"""
        vocab_data = {
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'word_freq': dict(self.word_freq),
            'min_freq': self.min_freq
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
    
    def load(self, filepath):
        """加载词汇表"""
        with open(filepath, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        self.word2idx = vocab_data['word2idx']
        self.idx2word = {int(k): v for k, v in vocab_data['idx2word'].items()}
        self.word_freq = Counter(vocab_data['word_freq'])
        self.min_freq = vocab_data['min_freq']
    
    @property
    def vocab_size(self):
        """词汇表大小"""
        return len(self.word2idx)
    
    @property
    def pad_idx(self):
        """PAD标记的索引"""
        return self.word2idx[self.PAD_TOKEN]
    
    @property
    def unk_idx(self):
        """UNK标记的索引"""
        return self.word2idx[self.UNK_TOKEN]
    
    @property
    def sos_idx(self):
        """SOS标记的索引"""
        return self.word2idx[self.SOS_TOKEN]
    
    @property
    def eos_idx(self):
        """EOS标记的索引"""
        return self.word2idx[self.EOS_TOKEN]
