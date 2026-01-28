"""
PatchTST Model
"""
from typing import List, Tuple, Union

import torch
import torch.nn as nn

from .config_PatchTST import PatchTSTConfig
from .layers_PatchTST import RevIN, PatchEmbedding, PositionalEncoding, TransformerEncoderLayer


class PatchTST(nn.Module):
    """
    PatchTST 模型

    核心設計:
    1. Patching: 聚合稀疏神經脈衝，提升語義密度
    2. Channel Independence: 規避不穩定的神經元間相關性
    3. RevIN: 自動適應分佈偏移
    """

    def __init__(self, config: PatchTSTConfig):
        super().__init__()
        self.config = config

        if config.revin:
            self.revin_layer = RevIN(1, affine=config.affine)

        self.patch_embedding = PatchEmbedding(
            patch_len=config.patch_len,
            stride=config.stride,
            d_model=config.d_model,
            dropout=config.dropout
        )

        self.pos_encoding = PositionalEncoding(
            d_model=config.d_model,
            max_len=config.n_patches + 100,
            dropout=config.dropout
        )

        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_ff=config.d_ff,
                dropout=config.dropout
            )
            for _ in range(config.e_layers)
        ])

        self.flatten_head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(config.n_patches * config.d_model, config.horizon),
        )

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size, seq_len, n_vars = x.shape

        x = x.permute(0, 2, 1).reshape(batch_size * n_vars, seq_len, 1)

        if self.config.revin:
            x = self.revin_layer(x, mode='norm')

        x = self.patch_embedding(x)
        x = self.pos_encoding(x)

        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)

        features = x

        x = self.flatten_head(x)
        x = x.unsqueeze(-1)

        if self.config.revin:
            x = self.revin_layer(x, mode='denorm')

        x = x.squeeze(-1).reshape(batch_size, n_vars, -1).permute(0, 2, 1)

        if return_features:
            return x, features
        return x


class MultiScaleEnsemble(nn.Module):
    """
    多尺度 Patching 集成

    使用不同的 patch_len 捕捉不同頻率成分:
    - 小 patch: 高頻 Spikes
    - 大 patch: 低頻振盪趨勢
    """

    def __init__(
        self,
        base_config: PatchTSTConfig,
        patch_sizes: List[int] = [4, 8, 16]
    ):
        super().__init__()
        self.models = nn.ModuleList()

        for patch_len in patch_sizes:
            config = PatchTSTConfig(
                input_size=base_config.input_size,
                horizon=base_config.horizon,
                patch_len=patch_len,
                stride=patch_len // 2,
                d_model=base_config.d_model,
                n_heads=base_config.n_heads,
                e_layers=base_config.e_layers,
                dropout=base_config.dropout,
                revin=base_config.revin,
                affine=base_config.affine,
                device=base_config.device
            )
            self.models.append(PatchTST(config))

        self.n_models = len(patch_sizes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        predictions = [model(x) for model in self.models]
        return torch.stack(predictions).mean(dim=0)
