#!/usr/bin/env python3
"""
高級時間序列預測模型
======================

實現最新的 SOTA 模型架構：
1. iTransformer - 反轉 Transformer（每個變量獨立處理）
2. TimesNet - 多周期時間序列建模
3. DLinear - 簡單但強大的線性基線
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


# ============================================================================
# iTransformer - Inverted Transformer
# ============================================================================

class iTransformer(nn.Module):
    """
    iTransformer - 反轉 Transformer 架構

    關鍵創新：
    - 將每個變量（神經元）視為一個 token
    - 在時間維度上進行嵌入
    - 在變量維度上進行自注意力

    非常適合多變量獨立預測的場景
    """

    def __init__(
        self,
        input_size: int = 96,
        horizon: int = 24,
        n_vars: int = 1,  # 將在運行時自動檢測
        d_model: int = 128,
        n_heads: int = 4,
        e_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.1,
        use_norm: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.horizon = horizon
        self.use_norm = use_norm

        # 變量嵌入 - 將每個變量的時間序列投影到 d_model 維度
        self.var_embedding = nn.Linear(input_size, d_model)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)

        # 預測頭
        self.projection = nn.Linear(d_model, horizon)

        if self.use_norm:
            self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, n_vars]

        Returns:
            [batch_size, horizon, n_vars]
        """
        batch_size, seq_len, n_vars = x.shape

        # 轉置: [batch, n_vars, seq_len]
        x = x.permute(0, 2, 1)

        # 變量嵌入: [batch, n_vars, d_model]
        x = self.var_embedding(x)

        if self.use_norm:
            x = self.norm(x)

        # Transformer: [batch, n_vars, d_model]
        x = self.encoder(x)

        # 投影到預測視距: [batch, n_vars, horizon]
        x = self.projection(x)

        # 轉置回: [batch, horizon, n_vars]
        x = x.permute(0, 2, 1)

        return x


# ============================================================================
# TimesNet - Temporal 2D-Variation Modeling
# ============================================================================

class Inception_Block_V1(nn.Module):
    """Inception 塊 - 多尺度卷積"""

    def __init__(self, in_channels: int, out_channels: int, num_kernels: int = 6):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels

        kernels = []
        for i in range(num_kernels):
            kernels.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=2*i+1, padding=i)
            )
        self.kernels = nn.ModuleList(kernels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for kernel in self.kernels:
            outputs.append(kernel(x))
        return torch.stack(outputs, dim=-1).mean(dim=-1)


class TimesBlock(nn.Module):
    """TimesNet 基礎塊"""

    def __init__(
        self,
        input_size: int,
        horizon: int,
        d_model: int = 64,
        d_ff: int = 128,
        top_k: int = 5,
        num_kernels: int = 6
    ):
        super().__init__()
        self.input_size = input_size
        self.horizon = horizon
        self.d_model = d_model
        self.top_k = top_k

        # FFT 用於頻率分析
        self.conv = nn.Sequential(
            Inception_Block_V1(d_model, d_ff, num_kernels=num_kernels),
            nn.GELU(),
            Inception_Block_V1(d_ff, d_model, num_kernels=num_kernels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N = x.shape

        # FFT 分析周期性
        x_freq = torch.fft.rfft(x, dim=1)
        freq_magnitude = torch.abs(x_freq).mean(dim=-1)

        # 選擇 top-k 頻率
        _, top_list = torch.topk(freq_magnitude, self.top_k, dim=1)

        x_out = []
        for i in range(self.top_k):
            # 提取周期
            period = (T // (top_list[:, i] + 1)).int()

            # 重塑為 2D
            period_max = period.max().item()
            if period_max == 0:
                continue

            # Padding 到周期長度
            pad_len = period_max - (T % period_max) if T % period_max != 0 else 0
            if pad_len > 0:
                x_padded = F.pad(x, (0, 0, 0, pad_len))
            else:
                x_padded = x

            # 重塑為 2D: [B, period, T/period, N]
            x_2d = x_padded.reshape(B, period_max, -1, N).permute(0, 3, 1, 2)

            # 2D 卷積
            x_conv = self.conv(x_2d)

            # 還原
            x_conv = x_conv.permute(0, 2, 3, 1).reshape(B, -1, N)
            if pad_len > 0:
                x_conv = x_conv[:, :T, :]

            x_out.append(x_conv)

        if len(x_out) == 0:
            return x

        return sum(x_out) / len(x_out) + x


class TimesNet(nn.Module):
    """
    TimesNet - 時序 2D 變化建模

    關鍵思想:
    - 將 1D 時間序列轉換為 2D 張量（基於周期性）
    - 使用 2D 卷積捕捉時序變化
    - 自適應聚合多個周期
    """

    def __init__(
        self,
        input_size: int = 96,
        horizon: int = 24,
        d_model: int = 64,
        d_ff: int = 128,
        num_blocks: int = 2,
        top_k: int = 5,
        num_kernels: int = 6,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_size = input_size
        self.horizon = horizon

        # 輸入嵌入
        self.embedding = nn.Linear(1, d_model)

        # TimesBlocks
        self.blocks = nn.ModuleList([
            TimesBlock(input_size, horizon, d_model, d_ff, top_k, num_kernels)
            for _ in range(num_blocks)
        ])

        # 層歸一化
        self.layer_norm = nn.LayerNorm(d_model)

        # 預測頭
        self.projection = nn.Linear(input_size, horizon)
        self.output_projection = nn.Linear(d_model, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, n_vars]

        Returns:
            [batch_size, horizon, n_vars]
        """
        B, T, N = x.shape

        # Channel independence: 處理每個變量
        x = x.permute(0, 2, 1).reshape(B * N, T, 1)

        # 嵌入
        x = self.embedding(x)  # [B*N, T, d_model]

        # TimesBlocks
        for block in self.blocks:
            x = block(x)
            x = self.dropout(self.layer_norm(x))

        # 投影到 horizon
        x = x.permute(0, 2, 1)  # [B*N, d_model, T]
        x = self.projection(x)  # [B*N, d_model, horizon]
        x = x.permute(0, 2, 1)  # [B*N, horizon, d_model]

        # 輸出投影
        x = self.output_projection(x)  # [B*N, horizon, 1]

        # 還原形狀
        x = x.reshape(B, N, self.horizon).permute(0, 2, 1)

        return x


# ============================================================================
# DLinear - 簡單但強大的線性模型
# ============================================================================

class MovingAvg(nn.Module):
    """移動平均 - 用於趨勢分解"""

    def __init__(self, kernel_size: int = 25, stride: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 填充
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)

        # 移動平均
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class SeriesDecomp(nn.Module):
    """序列分解 - 趨勢 + 季節性"""

    def __init__(self, kernel_size: int = 25):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend


class DLinear(nn.Module):
    """
    DLinear - 分解線性模型

    核心思想：
    - 將時間序列分解為趨勢 + 季節性
    - 分別用線性層預測
    - 簡單但極其有效的基線
    """

    def __init__(
        self,
        input_size: int = 96,
        horizon: int = 24,
        kernel_size: int = 25,
        individual: bool = False
    ):
        super().__init__()
        self.input_size = input_size
        self.horizon = horizon
        self.individual = individual

        # 序列分解
        self.decomposition = SeriesDecomp(kernel_size)

        if individual:
            # 每個變量獨立的線性層
            self.seasonal_linear = nn.ModuleList([
                nn.Linear(input_size, horizon) for _ in range(100)  # 最多支援 100 個變量
            ])
            self.trend_linear = nn.ModuleList([
                nn.Linear(input_size, horizon) for _ in range(100)
            ])
        else:
            # 共享線性層
            self.seasonal_linear = nn.Linear(input_size, horizon)
            self.trend_linear = nn.Linear(input_size, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, n_vars]

        Returns:
            [batch_size, horizon, n_vars]
        """
        # 分解
        seasonal, trend = self.decomposition(x)

        # 預測
        if self.individual:
            seasonal_out = []
            trend_out = []
            for i in range(x.shape[2]):
                seasonal_out.append(self.seasonal_linear[i](seasonal[:, :, i]))
                trend_out.append(self.trend_linear[i](trend[:, :, i]))
            seasonal_out = torch.stack(seasonal_out, dim=2)
            trend_out = torch.stack(trend_out, dim=2)
        else:
            seasonal_out = self.seasonal_linear(seasonal.permute(0, 2, 1)).permute(0, 2, 1)
            trend_out = self.trend_linear(trend.permute(0, 2, 1)).permute(0, 2, 1)

        return seasonal_out + trend_out


# ============================================================================
# 模型工廠
# ============================================================================

class ModelFactory:
    """統一的模型創建接口"""

    @staticmethod
    def create_model(
        model_name: str,
        input_size: int = 96,
        horizon: int = 24,
        **kwargs
    ) -> nn.Module:
        """
        創建模型

        Args:
            model_name: 模型名稱 ('itransformer', 'timesnet', 'dlinear')
            input_size: 輸入序列長度
            horizon: 預測長度
            **kwargs: 其他模型參數
        """
        if model_name.lower() == 'itransformer':
            return iTransformer(input_size=input_size, horizon=horizon, **kwargs)
        elif model_name.lower() == 'timesnet':
            return TimesNet(input_size=input_size, horizon=horizon, **kwargs)
        elif model_name.lower() == 'dlinear':
            return DLinear(input_size=input_size, horizon=horizon, **kwargs)
        else:
            raise ValueError(f"Unknown model: {model_name}")
