"""
Transformer解码器
用于Model C, D, E (修复版)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from ..layers.attention import MultiHeadAttention, PositionalEncoding


class TransformerDecoderLayer(nn.Module):
    """Transformer解码器层"""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, encoder_output, tgt_mask=None, src_mask=None):
        # 自注意力
        attn_output, _ = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 交叉注意力
        cross_attn_output, _ = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        # 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x


class TransformerDecoder(nn.Module):
    """Transformer解码器"""
    
    def __init__(self, vocab_size, d_model=512, num_heads=8,
                num_encoder_layers=6, num_decoder_layers=6,
                d_ff=2048, dropout=0.1, max_len=100,
                use_image_pos_encoding: bool = False,
                image_max_len: int = 100,
                skip_encoder_layers: bool = False):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # 词嵌入
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.use_image_pos_encoding = use_image_pos_encoding
        if self.use_image_pos_encoding:
            self.image_pos_encoding = PositionalEncoding(d_model, image_max_len)
        self.skip_encoder_layers = skip_encoder_layers
        
        # 编码器层（处理视觉特征）
        self.encoder_layers = nn.ModuleList([
            nn.ModuleDict({
                'self_attn': MultiHeadAttention(d_model, num_heads, dropout),
                'ff': nn.Sequential(
                    nn.Linear(d_model, d_ff),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_ff, d_model)
                ),
                'norm1': nn.LayerNorm(d_model),
                'norm2': nn.LayerNorm(d_model),
                'dropout': nn.Dropout(dropout)
            })
            for _ in range(num_encoder_layers)
        ])
        
        # 解码器层
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])
        
        # 输出层
        self.output_projection = nn.Linear(d_model, vocab_size)
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, visual_features, captions=None, tgt_mask=None, src_mask=None):
        x_vis = visual_features
        if self.use_image_pos_encoding and x_vis is not None:
            x_vis = x_vis.permute(1, 0, 2)
            x_vis = self.image_pos_encoding(x_vis)
            x_vis = x_vis.permute(1, 0, 2)

        if not self.skip_encoder_layers:
            for layer in self.encoder_layers:
                attn_output, _ = layer['self_attn'](x_vis, x_vis, x_vis, src_mask)
                x_vis = layer['norm1'](x_vis + layer['dropout'](attn_output))
                ff_output = layer['ff'](x_vis)
                x_vis = layer['norm2'](x_vis + layer['dropout'](ff_output))
        
        if captions is not None:
            x = self.embedding(captions) * math.sqrt(self.d_model)
            x = x.transpose(0, 1)
            x = self.pos_encoding(x)
            x = x.transpose(0, 1)
            x = self.dropout(x)
            
            if tgt_mask is None:
                seq_len = captions.size(1)
                tgt_mask = self._generate_square_subsequent_mask(seq_len).to(captions.device)
            
            for layer in self.decoder_layers:
                x = layer(x, x_vis, tgt_mask, src_mask)
            
            output = self.output_projection(x)
            return output
        else:
            return x_vis
    
    def generate(
        self,
        visual_features,
        vocab=None,
        max_length=50,
        strategy='greedy',
        beam_size=3,
        temperature=1.0,
        top_k=5,
        top_p=0.9,
        no_repeat_ngram_size: int = 0,
        min_length: int = 0,
        length_penalty: float = 1.0,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        batch_size = visual_features.size(0)
        device = visual_features.device
        
        x_vis = visual_features
        for layer in self.encoder_layers:
            attn_output, _ = layer['self_attn'](x_vis, x_vis, x_vis, None)
            x_vis = layer['norm1'](x_vis + layer['dropout'](attn_output))
            ff_output = layer['ff'](x_vis)
            x_vis = layer['norm2'](x_vis + layer['dropout'](ff_output))
        
        sequences = []
        sequences_log_probs = []

        bos_idx = vocab.word2idx.get('<bos>', 1)
        pad_idx = vocab.word2idx.get('<pad>', 0)
        sos_idx = vocab.word2idx.get('<sos>', 1)
        eos_idx = vocab.word2idx.get('<eos>', 3)

        for b in range(batch_size):
            src_feats = x_vis[b:b+1]

            # -------------------------------------------------------
            # 策略 1: Sampling (保持原样，略作优化)
            # -------------------------------------------------------
            if strategy == 'sampling':
                cur_input = torch.tensor([bos_idx], device=device).unsqueeze(0)
                decoder_input = self.embedding(cur_input) * math.sqrt(self.d_model)
                decoder_input = decoder_input.transpose(0, 1)
                decoder_input = self.pos_encoding(decoder_input)
                decoder_input = decoder_input.transpose(0, 1)
                gen = []
                logp_list = []
                for step in range(max_length):
                    tgt_mask = self._generate_square_subsequent_mask(decoder_input.size(1)).to(device)
                    x = decoder_input
                    for layer in self.decoder_layers:
                        x = layer(x, src_feats, tgt_mask, None)
                    logits = self.output_projection(x[:, -1, :]) / temperature
                    
                    if logits.dim() == 1:
                        logits = logits.unsqueeze(0)
                    logits[0, pad_idx] = float('-inf')
                    logits[0, sos_idx] = float('-inf')
                    # 如果未达到最短长度，禁止生成 EOS（sampling）
                    if min_length and step < min_length:
                        logits[0, eos_idx] = float('-inf')
                    
                    probs = F.softmax(logits, dim=-1)
                    if top_k > 0:
                        top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)
                        probs = torch.zeros_like(probs)
                        probs.scatter_(1, top_k_indices, top_k_probs)
                        probs = probs / probs.sum(dim=-1, keepdim=True)
                    dist = torch.distributions.Categorical(probs)
                    pred = dist.sample()
                    gen.append(pred)
                    logp_list.append(dist.log_prob(pred))
                    if pred.item() == eos_idx:
                        break
                    next_embed = self.embedding(pred.unsqueeze(1)) * math.sqrt(self.d_model)
                    pos = self.pos_encoding.pe[step + 1: step + 2]
                    pos = pos.permute(1, 0, 2).expand(1, 1, -1)
                    next_embed = next_embed + pos
                    decoder_input = torch.cat([decoder_input, next_embed], dim=1)
                
                if gen:
                    seq = torch.stack(gen, dim=1)
                    sequences.append(seq)
                    sequences_log_probs.append(torch.stack(logp_list, dim=1))
                else:
                    sequences.append(torch.zeros((1, 0), dtype=torch.long, device=device))
                    sequences_log_probs.append(torch.zeros((1, 0), device=device))
                continue

            # -------------------------------------------------------
            # 策略 2: Beam Search (已修复重复代码和性能问题)
            # -------------------------------------------------------
            if strategy == 'beam_search' and beam_size > 1:
                beams = [([bos_idx], 0.0, False)]
                for step in range(max_length):
                    candidates = []
                    for tokens, score, finished in beams:
                        if finished:
                            candidates.append((tokens, score, True))
                            continue
                        
                        cur_seq = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
                        decoder_input = self.embedding(cur_seq) * math.sqrt(self.d_model)
                        decoder_input = decoder_input.transpose(0, 1)
                        decoder_input = self.pos_encoding(decoder_input)
                        decoder_input = decoder_input.transpose(0, 1)
                        tgt_mask = self._generate_square_subsequent_mask(decoder_input.size(1)).to(device)
                        
                        x = decoder_input
                        for layer in self.decoder_layers:
                            x = layer(x, src_feats, tgt_mask, None)
                        
                        logits = self.output_projection(x[:, -1, :]) / temperature
                        logits[0, pad_idx] = float('-inf')
                        logits[0, sos_idx] = float('-inf')

                        # 如果未达到最短长度，禁止生成 EOS
                        if min_length and step < min_length:
                            if logits.dim() == 1:
                                logits = logits.unsqueeze(0)
                            logits[0, eos_idx] = float('-inf')
                        # 计算 Log Softmax (已修复重复代码)
                        log_probs = F.log_softmax(logits, dim=-1)
                        if log_probs.dim() == 2 and log_probs.size(0) == 1:
                            log_probs = log_probs.squeeze(0)

                        # 应用 no_repeat_ngram_size (已优化性能)
                        if no_repeat_ngram_size > 0:
                            # 只需要检查当前序列是否会产生重复 n-gram
                            # 我们不遍历词表，而是遍历历史序列
                            if len(tokens) >= no_repeat_ngram_size - 1:
                                prefix_len = no_repeat_ngram_size - 1
                                # 获取当前想要延续的 "前缀" (即过去 N-1 个词)
                                current_prefix = tuple(tokens[-prefix_len:]) if prefix_len > 0 else ()
                                
                                # 遍历整个历史，看这个前缀以前出现过没有
                                # 只需要遍历到 len(tokens) - prefix_len，因为我们要找的是前缀后面的那个词
                                for i in range(len(tokens) - prefix_len):
                                    # 检查历史中的窗口
                                    window = tuple(tokens[i : i + prefix_len]) if prefix_len > 0 else ()
                                    if window == current_prefix:
                                        # 如果匹配，说明如果选了 tokens[i+prefix_len]，就会构成重复 n-gram
                                        banned_token_id = tokens[i + prefix_len]
                                        log_probs[banned_token_id] = float('-inf')

                        # 选出 Top-K
                        topk_logps, topk_ids = torch.topk(log_probs, min(beam_size, log_probs.size(0)))
                        for k_logp, k_id in zip(topk_logps.tolist(), topk_ids.tolist()):
                            new_tokens = tokens + [int(k_id)]
                            new_score = score + float(k_logp)
                            finished_flag = (int(k_id) == eos_idx)
                            candidates.append((new_tokens, new_score, finished_flag))
                    
                    # 根据 length_penalty 对候选项评分并选出前 beam_size 个
                    def norm_score(item):
                        toks, sc, fin = item
                        length = max(1, len(toks))
                        return sc / (length ** float(length_penalty))

                    candidates = sorted(candidates, key=lambda x: norm_score(x), reverse=True)[:beam_size]
                    beams = candidates
                    if all(f for (_, _, f) in beams):
                        break
                
                # 最终选择时也按归一化得分排序
                beams = sorted(beams, key=lambda x: (x[1] / (max(1, len(x[0])) ** float(length_penalty))), reverse=True)
                best_tokens = beams[0][0]
                if len(best_tokens) > 0 and best_tokens[0] == bos_idx:
                    best_tokens = best_tokens[1:]
                seq = torch.tensor(best_tokens, dtype=torch.long, device=device).unsqueeze(0)
                sequences.append(seq)
                sequences_log_probs.append(torch.tensor([beams[0][1]], device=device).unsqueeze(0))
                continue

            # -------------------------------------------------------
            # 策略 3: Greedy (已优化 N-gram 性能)
            # -------------------------------------------------------
            cur_input = torch.tensor([bos_idx], device=device).unsqueeze(0)
            decoder_input = self.embedding(cur_input) * math.sqrt(self.d_model)
            decoder_input = decoder_input.transpose(0, 1)
            decoder_input = self.pos_encoding(decoder_input)
            decoder_input = decoder_input.transpose(0, 1)
            gen = []
            logp_list = []
            
            for step in range(max_length):
                tgt_mask = self._generate_square_subsequent_mask(decoder_input.size(1)).to(device)
                x = decoder_input
                for layer in self.decoder_layers:
                    x = layer(x, src_feats, tgt_mask, None)
                
                logits = self.output_projection(x[:, -1, :]) / temperature
                if logits.dim() == 1: logits = logits.unsqueeze(0)
                logits[0, pad_idx] = float('-inf')
                logits[0, sos_idx] = float('-inf')

                # 应用 no_repeat_ngram_size (已优化)
                if no_repeat_ngram_size > 0 and len(gen) >= no_repeat_ngram_size - 1:
                    prefix_len = no_repeat_ngram_size - 1
                    current_gen_ids = [t.item() for t in gen]
                    current_prefix = tuple(current_gen_ids[-prefix_len:]) if prefix_len > 0 else ()
                    
                    for i in range(len(current_gen_ids) - prefix_len):
                        window = tuple(current_gen_ids[i : i + prefix_len]) if prefix_len > 0 else ()
                        if window == current_prefix:
                            banned_id = current_gen_ids[i + prefix_len]
                            logits[0, banned_id] = float('-inf')

                # 如果未达到最短长度，禁止生成 EOS
                if min_length and step < min_length:
                    if logits.dim() == 1:
                        logits = logits.unsqueeze(0)
                    logits[0, eos_idx] = float('-inf')

                pred = logits.argmax(1)
                probs = F.softmax(logits, dim=-1)
                logp = torch.log(probs.gather(1, pred.unsqueeze(1)).squeeze(1) + 1e-10)
                gen.append(pred)
                logp_list.append(logp)
                
                if pred.item() == eos_idx:
                    break
                    
                next_embed = self.embedding(pred.unsqueeze(1)) * math.sqrt(self.d_model)
                pos = self.pos_encoding.pe[step + 1: step + 2]
                pos = pos.permute(1, 0, 2).expand(1, 1, -1)
                next_embed = next_embed + pos
                decoder_input = torch.cat([decoder_input, next_embed], dim=1)
                
            if gen:
                seq = torch.stack(gen, dim=1)
                sequences.append(seq)
                sequences_log_probs.append(torch.stack(logp_list, dim=1))
            else:
                sequences.append(torch.zeros((1, 0), dtype=torch.long, device=device))
                sequences_log_probs.append(torch.zeros((1, 0), device=device))

        # Pad results
        max_len_out = max([s.size(1) for s in sequences]) if sequences else 0
        out_seqs = torch.zeros((batch_size, max_len_out), dtype=torch.long, device=device)
        out_logps = []
        for i, s in enumerate(sequences):
            if s.size(1) > 0:
                out_seqs[i, :s.size(1)] = s
            out_logps.append(sequences_log_probs[i])
            
        return out_seqs, None
    
    def _generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask