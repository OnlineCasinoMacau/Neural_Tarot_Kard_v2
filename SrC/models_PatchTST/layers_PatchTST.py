"""
PatchTST Layers - RevIN, Patch Embedding, Positional Encoding, Transformer Encoder
"""
import numpy as np
import torch
import torch.nn as nn


class RevIN(nn.Module):
    """
    可逆實例歸一化 - 處理 OOD 漂移的核心組件

    數學原理:
    1. 歸一化: x_norm = (x - μ) / (σ + ε)
    2. 模型推理
    3. 反歸一化: y = y_norm * σ + μ
    """

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = False):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor, mode: str = 'norm') -> torch.Tensor:
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        return x

    def _get_statistics(self, x: torch.Tensor):
        self.mean = x.mean(dim=1, keepdim=True).detach()
        self.std = x.std(dim=1, keepdim=True).detach() + self.eps

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.std
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.affine:
            x = (x - self.affine_bias) / self.affine_weight
        x = x * self.std + self.mean
        return x


class PatchEmbedding(nn.Module):
    """Patch 嵌入層 - 將時間序列分割並投影到高維空間"""

    def __init__(
        self,
        patch_len: int,
        stride: int,
        d_model: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.projection = nn.Linear(patch_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.squeeze(-1)
        x = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        x = self.projection(x)
        x = self.dropout(x)
        return x


class PositionalEncoding(nn.Module):
    """位置編碼"""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    """Transformer Encoder 層"""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x
