"""
PatchTST Configuration
"""
from typing import Dict
import torch


class PatchTSTConfig:
    """PatchTST 模型配置"""

    def __init__(
        self,
        # 序列參數
        input_size: int = 96,
        horizon: int = 24,

        # Patching 參數
        patch_len: int = 8,
        stride: int = 4,

        # Transformer 參數
        d_model: int = 128,
        n_heads: int = 4,
        e_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.2,

        # RevIN 參數
        revin: bool = True,
        affine: bool = False,

        # 訓練參數
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs: int = 100,
        early_stop_patience: int = 10,
        gradient_clip: float = 1.0,

        # 損失函數權重
        mse_weight: float = 1.0,
        mae_weight: float = 0.5,

        # 數據預處理
        gaussian_sigma: float = 0.05,
        use_sqrt_transform: bool = True,

        # 設備
        device: str = 'auto',
    ):
        self.input_size = input_size
        self.horizon = horizon
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.revin = revin
        self.affine = affine
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.early_stop_patience = early_stop_patience
        self.gradient_clip = gradient_clip
        self.mse_weight = mse_weight
        self.mae_weight = mae_weight
        self.gaussian_sigma = gaussian_sigma
        self.use_sqrt_transform = use_sqrt_transform

        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.n_patches = (self.input_size - self.patch_len) // self.stride + 1

    def to_dict(self) -> Dict:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'PatchTSTConfig':
        return cls(**config_dict)

    @classmethod
    def for_short_sequence(cls, seq_len: int, horizon: int = 24) -> 'PatchTSTConfig':
        """為短序列自動調整配置"""
        if seq_len < 12:
            patch_len, stride = 4, 1
        elif seq_len < 24:
            patch_len, stride = 4, 2
        elif seq_len < 96:
            patch_len, stride = 8, 4
        else:
            patch_len, stride = 16, 8

        return cls(
            input_size=seq_len,
            horizon=horizon,
            patch_len=patch_len,
            stride=stride
        )
