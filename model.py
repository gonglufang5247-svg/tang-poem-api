"""You should not need to change this file. The GPT is built out of your six functions in student.py.

Defines the Transformer: Block (self-attention + MLP) and GPT (embeddings, blocks, output head).
"""

import math

import torch
import torch.nn as nn

import student


class Block(nn.Module):
    def __init__(self, C, H, dropout):
        super().__init__()
        self.H = H
        self.d_head = C // H
        self.ln1, self.ln2 = nn.LayerNorm(C), nn.LayerNorm(C)
        self.qkv = nn.Linear(C, 3 * C)
        self.proj = nn.Linear(C, C)
        self.mlp = nn.Sequential(nn.Linear(C, 4 * C), nn.GELU(), nn.Linear(4 * C, C))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(C, dim=2)
        q = q.view(B, T, self.H, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.H, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.H, self.d_head).transpose(1, 2)
        mask = student.causal_mask(T).to(x.device)
        a = student.attention(q, k, v, mask)
        a = a.transpose(1, 2).reshape(B, T, C)
        x = x + self.drop(self.proj(a))
        x = x + self.drop(self.mlp(self.ln2(x)))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, cfg):
        super().__init__()
        C, T = cfg["n_embd"], cfg["block_size"]
        self.block_size = T
        self.tok = nn.Embedding(vocab_size, C)
        self.pos = nn.Embedding(T, C)
        self.blocks = nn.Sequential(
            *[Block(C, cfg["n_head"], cfg.get("dropout", 0.2)) for _ in range(cfg["n_layer"])]
        )
        self.ln = nn.LayerNorm(C)
        self.head = nn.Linear(C, vocab_size, bias=False)
        self.head.weight = self.tok.weight
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx):
        x = student.add_positions(self.tok(idx), self.pos.weight)
        x = self.blocks(x)
        return self.head(self.ln(x))
