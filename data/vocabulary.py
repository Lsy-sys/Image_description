"""
词汇表构建和管理
"""

import json
import pickle
from collections import Counter
import numpy as np
import re


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
        # 基于正则的简易分词并清洗标点：
        # - 小写化
        # - 提取字母/数字序列，保留常见缩写的撇号（如 don't）
        # 这可以避免将 "skirt."、"skirt," 等视为不同词。
        if not isinstance(text, str):
            return []
        text_lower = text.lower()
        # 匹配字母数字序列，允许中间带撇号或连字符
        tokens = re.findall(r"[a-z0-9]+(?:['-][a-z0-9]+)*", text_lower)
        return tokens
    
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

    # -------- 新增便捷构建方法 --------
    @classmethod
    def from_captions(cls, data_dir: str, min_freq: int = 5, vocab_path: str = None):
        """
        从数据集目录构建并返回 Vocabulary 实例。
        如果 vocab_path 存在，则直接加载，避免重复构建。
        Args:
            data_dir: 数据集根目录（需包含 train/val/test 列表和 captions 目录）
            min_freq: 最小词频阈值（仅在需要构建时使用）
            vocab_path: 词表文件路径，如果文件存在则直接加载，否则构建后保存
        """
        import os
        
        # 如果提供了 vocab_path 且文件存在，直接加载
        if vocab_path and os.path.exists(vocab_path):
            print(f"从文件加载词汇表: {vocab_path}")
            vocab = cls.load(vocab_path)
            print(f"词汇表大小: {len(vocab)}")
            return vocab
        
        # 否则，从数据集构建
        print("从数据集构建词汇表...")
        vocab = cls(min_freq=min_freq)
        vocab.build_vocab_from_dataset(data_dir)
        
        # 如果提供了 vocab_path，保存构建好的词表
        if vocab_path:
            os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
            vocab.save(vocab_path)
            print(f"词汇表已保存到: {vocab_path}")
        
        return vocab

    @classmethod
    def load(cls, filepath: str):
        """
        从文件加载 Vocabulary 实例
        """
        vocab = cls()
        with open(filepath, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        vocab.word2idx = vocab_data['word2idx']
        vocab.idx2word = {int(k): v for k, v in vocab_data['idx2word'].items()}
        vocab.word_freq = Counter(vocab_data['word_freq'])
        vocab.min_freq = vocab_data['min_freq']
        return vocab