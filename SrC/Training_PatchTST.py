#!/usr/bin/env python3
"""
NSF HDR A3D3 Neural Forecasting Competition - PatchTST Training Script
======================================================================

訓練腳本 - 專注於數據處理與訓練邏輯
模型架構定義請見 models/ 目錄
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict, List, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

from models_PatchTST import PatchTSTConfig, PatchTST

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# 數據預處理函數
# ============================================================================

def gaussian_smooth(data: np.ndarray, sigma: float, dt: float = 0.01) -> np.ndarray:
    """
    高斯核平滑 - 將離散脈衝轉為連續放電率

    Args:
        data: 原始信號 (time, channels)
        sigma: 高斯核標準差 (秒)
        dt: 時間步長 (秒)
    """
    from scipy.ndimage import gaussian_filter1d
    sigma_samples = sigma / dt
    smoothed = gaussian_filter1d(
        data, sigma=sigma_samples, axis=0, mode='constant', cval=0
    )
    return smoothed


def sqrt_transform(data: np.ndarray) -> np.ndarray:
    """平方根變換 (Anscombe Transform) - 穩定泊松方差"""
    # Ensure non-negative values before sqrt to prevent NaN
    data_shifted = data + 3/8
    data_shifted = np.maximum(data_shifted, 0)  # Clip negative values to 0
    return np.sqrt(data_shifted)


def inverse_sqrt_transform(data: np.ndarray) -> np.ndarray:
    """反向平方根變換"""
    return np.maximum(data ** 2 - 3/8, 0)


def pad_sequence(
    x: torch.Tensor,
    target_len: int,
    mode: str = 'edge'
) -> torch.Tensor:
    """序列填充 - 處理短序列輸入"""
    current_len = x.shape[1]
    if current_len >= target_len:
        return x[:, -target_len:, :]

    pad_len = target_len - current_len

    if mode == 'zero':
        pad = torch.zeros(x.shape[0], pad_len, x.shape[2], device=x.device)
    elif mode == 'edge':
        pad = x[:, :1, :].repeat(1, pad_len, 1)
    elif mode == 'reflect':
        pad = x[:, :pad_len, :].flip(dims=[1])
    else:
        raise ValueError(f"Unknown padding mode: {mode}")

    return torch.cat([pad, x], dim=1)


# ============================================================================
# 數據集
# ============================================================================

class NeuralDataset(Dataset):
    """神經訊號數據集"""

    def __init__(
        self,
        data: np.ndarray,
        input_size: int,
        horizon: int,
        stride: int = 1,
        transform: bool = True,
        gaussian_sigma: float = 0.05,
        dt: float = 0.01
    ):
        self.input_size = input_size
        self.horizon = horizon
        self.stride = stride

        if transform:
            data = gaussian_smooth(data, sigma=gaussian_sigma, dt=dt)
            data = sqrt_transform(data)

            # Validate data after transformation
            if np.isnan(data).any():
                raise ValueError(f"NaN values detected after preprocessing! Check data range.")
            if np.isinf(data).any():
                raise ValueError(f"Inf values detected after preprocessing! Check data range.")

        self.data = torch.FloatTensor(data)
        self.n_samples = (len(data) - input_size - horizon) // stride + 1

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.stride
        x = self.data[start:start + self.input_size]
        y = self.data[start + self.input_size:start + self.input_size + self.horizon]
        return x, y


# ============================================================================
# 損失函數
# ============================================================================

class CombinedLoss(nn.Module):
    """組合損失函數: MSE + MAE"""

    def __init__(self, mse_weight: float = 1.0, mae_weight: float = 0.5):
        super().__init__()
        self.mse_weight = mse_weight
        self.mae_weight = mae_weight
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.mse_weight * self.mse(pred, target) + \
               self.mae_weight * self.mae(pred, target)


# ============================================================================
# 訓練器
# ============================================================================

class PatchTSTTrainer:
    """PatchTST 訓練器"""

    def __init__(
        self,
        model: PatchTST,
        config: PatchTSTConfig,
        save_dir: str = './checkpoints'
    ):
        self.model = model.to(config.device)
        self.config = config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        self.criterion = CombinedLoss(
            mse_weight=config.mse_weight,
            mae_weight=config.mae_weight
        )

        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_mse': [],
            'best_val_loss': float('inf'),
            'best_epoch': 0
        }

    def train_epoch(self, dataloader: DataLoader, epoch: int = 0) -> float:
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(
            dataloader,
            desc=f"Epoch {epoch+1}/{self.config.max_epochs} [Train]",
            leave=False
        )

        for batch_idx, (x, y) in enumerate(pbar):
            x = x.to(self.config.device)
            y = y.to(self.config.device)

            pred = self.model(x)
            loss = self.criterion(pred, y)

            self.optimizer.zero_grad()
            loss.backward()

            if self.config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip
                )

            self.optimizer.step()
            total_loss += loss.item()

            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        return total_loss / len(dataloader)

    @torch.no_grad()
    def validate(self, dataloader: DataLoader, show_progress: bool = False) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        total_mse = 0.0
        mse_fn = nn.MSELoss()

        if show_progress:
            dataloader = tqdm(dataloader, desc="[Validate]", leave=False)

        for x, y in dataloader:
            x = x.to(self.config.device)
            y = y.to(self.config.device)

            pred = self.model(x)
            loss = self.criterion(pred, y)
            mse = mse_fn(pred, y)

            total_loss += loss.item()
            total_mse += mse.item()

        return total_loss / len(dataloader), total_mse / len(dataloader)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        scheduler: Optional[object] = None
    ) -> Dict:
        logger.info(f"開始訓練 - 設備: {self.config.device}")
        logger.info(f"模型參數量: {sum(p.numel() for p in self.model.parameters()):,}")

        if scheduler is None:
            scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=10,
                T_mult=2,
                eta_min=1e-6
            )

        patience_counter = 0

        # Epoch progress bar
        epoch_pbar = tqdm(
            range(self.config.max_epochs),
            desc="Training",
            unit="epoch"
        )

        for epoch in epoch_pbar:
            train_loss = self.train_epoch(train_loader, epoch)
            self.history['train_loss'].append(train_loss)

            if val_loader is not None:
                val_loss, val_mse = self.validate(val_loader, show_progress=False)
                self.history['val_loss'].append(val_loss)
                self.history['val_mse'].append(val_mse)

                # Update epoch progress bar
                epoch_pbar.set_postfix({
                    'train_loss': f'{train_loss:.4f}',
                    'val_loss': f'{val_loss:.4f}',
                    'val_mse': f'{val_mse:.4f}'
                })

                if val_loss < self.history['best_val_loss']:
                    self.history['best_val_loss'] = val_loss
                    self.history['best_epoch'] = epoch
                    patience_counter = 0
                    self.save_model('best_model.pt', silent=True)
                else:
                    patience_counter += 1

                if patience_counter >= self.config.early_stop_patience:
                    logger.info(f"\nEarly stopping at epoch {epoch+1}")
                    break
            else:
                epoch_pbar.set_postfix({'train_loss': f'{train_loss:.4f}'})

            scheduler.step()

        self.save_model('final_model.pt', silent=True)
        logger.info(f"\n訓練完成 - 最佳 Epoch: {self.history['best_epoch']+1}")

        return self.history

    def save_model(self, filename: str, silent: bool = False):
        path = self.save_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.to_dict(),
            'history': self.history
        }, path)
        if not silent:
            logger.info(f"模型已保存至: {path}")

    def load_model(self, filename: str):
        path = self.save_dir / filename
        checkpoint = torch.load(path, map_location=self.config.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        logger.info(f"模型已從 {path} 加載")


# ============================================================================
# 推理接口
# ============================================================================

class PatchTSTPredictor:
    """推理接口 - 用於 Codabench 提交"""

    def __init__(self, model: PatchTST, config: PatchTSTConfig):
        self.model = model
        self.config = config
        self.model.eval()

    @classmethod
    def load(cls, path: str, device: str = 'auto') -> 'PatchTSTPredictor':
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        checkpoint = torch.load(path, map_location=device)
        config = PatchTSTConfig.from_dict(checkpoint['config'])
        config.device = device

        model = PatchTST(config).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        return cls(model, config)

    @torch.no_grad()
    def predict(
        self,
        x: Union[np.ndarray, torch.Tensor],
        preprocess: bool = True
    ) -> np.ndarray:
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x)

        if x.dim() == 2:
            x = x.unsqueeze(0)

        if preprocess:
            x_np = x.numpy()
            x_np = gaussian_smooth(x_np[0], sigma=self.config.gaussian_sigma)
            x_np = sqrt_transform(x_np)
            x = torch.FloatTensor(x_np).unsqueeze(0)

        if x.shape[1] < self.config.input_size:
            x = pad_sequence(x, self.config.input_size, mode='edge')
        elif x.shape[1] > self.config.input_size:
            x = x[:, -self.config.input_size:, :]

        x = x.to(self.config.device)
        pred = self.model(x)

        pred_np = pred.cpu().numpy()
        if preprocess:
            pred_np = inverse_sqrt_transform(pred_np)

        return pred_np.squeeze(0)


# ============================================================================
# 數據加載函數
# ============================================================================

def load_npz_data(file_paths: List[str]) -> np.ndarray:
    """
    加載並合併多個 npz 文件的數據

    Args:
        file_paths: npz 文件路徑列表

    Returns:
        合併後的數據數組，形狀為 (time_steps, n_channels)
    """
    all_data = []

    for path in file_paths:
        logger.info(f"加載數據: {path}")
        with np.load(path) as npz:
            keys = list(npz.keys())
            logger.info(f"  可用 keys: {keys}")

            if 'data' in keys:
                data = npz['data']
            elif 'arr_0' in keys:
                data = npz['arr_0']
            else:
                data = npz[keys[0]]

            logger.info(f"  原始數據形狀: {data.shape}")

            # 確保數據是 2D: (time_steps, n_channels)
            if data.ndim == 1:
                # (time_steps,) -> (time_steps, 1)
                data = data.reshape(-1, 1)
            elif data.ndim == 3:
                # (n_trials, time_steps, n_channels) -> (n_trials * time_steps, n_channels)
                n_trials, time_steps, n_channels = data.shape
                data = data.reshape(-1, n_channels)
                logger.info(f"  重塑 3D -> 2D: {data.shape}")
            elif data.ndim == 4:
                # (n_sessions, n_trials, time_steps, n_channels) -> flatten
                data = data.reshape(-1, data.shape[-1])
                logger.info(f"  重塑 4D -> 2D: {data.shape}")
            elif data.ndim != 2:
                raise ValueError(f"不支援的數據維度: {data.ndim}")

            logger.info(f"  最終數據形狀: {data.shape}")
            all_data.append(data)

    if len(all_data) == 1:
        return all_data[0]
    else:
        return np.concatenate(all_data, axis=0)


def train_on_data(
    data_paths: List[str],
    config: PatchTSTConfig,
    save_dir: str,
    val_split: float = 0.2
) -> Tuple[PatchTST, Dict]:
    """
    在指定數據上訓練模型

    Args:
        data_paths: 數據文件路徑列表
        config: 模型配置
        save_dir: 模型保存目錄
        val_split: 驗證集比例

    Returns:
        訓練好的模型和訓練歷史
    """
    logger.info("=" * 60)
    logger.info("NSF HDR A3D3 Neural Forecasting - PatchTST Training")
    logger.info("=" * 60)

    # 加載數據
    logger.info("\n[Step 1] 加載數據...")
    data = load_npz_data(data_paths)

    # 時間序列劃分
    split_idx = int(len(data) * (1 - val_split))
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    logger.info(f"訓練集: {train_data.shape}, 驗證集: {val_data.shape}")

    # 創建數據集
    train_dataset = NeuralDataset(
        train_data,
        input_size=config.input_size,
        horizon=config.horizon,
        gaussian_sigma=config.gaussian_sigma,
        transform=True
    )

    val_dataset = NeuralDataset(
        val_data,
        input_size=config.input_size,
        horizon=config.horizon,
        gaussian_sigma=config.gaussian_sigma,
        transform=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if config.device == 'cuda' else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0
    )

    logger.info(f"訓練樣本數: {len(train_dataset)}, 驗證樣本數: {len(val_dataset)}")

    # 創建模型
    logger.info("\n[Step 2] 創建模型...")
    model = PatchTST(config)
    logger.info(f"配置: {json.dumps(config.to_dict(), indent=2)}")

    # 訓練
    logger.info("\n[Step 3] 開始訓練...")
    trainer = PatchTSTTrainer(
        model=model,
        config=config,
        save_dir=save_dir
    )

    history = trainer.fit(train_loader, val_loader)

    # 最終評估
    logger.info("\n[Step 4] 最終評估...")
    best_model_path = Path(save_dir) / 'best_model.pt'
    if best_model_path.exists():
        trainer.load_model('best_model.pt')
        final_loss, final_mse = trainer.validate(val_loader)
        logger.info(f"最終驗證 MSE: {final_mse:.6f}")
    else:
        logger.warning("best_model.pt not found, using current model state")
        final_loss, final_mse = trainer.validate(val_loader)
        logger.info(f"最終驗證 MSE: {final_mse:.6f}")

    # 保存部署包
    logger.info("\n[Step 5] 保存模型...")
    deployment_package = {
        'model_state_dict': model.state_dict(),
        'config': config.to_dict(),
        'history': history,
        'final_mse': final_mse,
        'timestamp': datetime.now().isoformat(),
        'data_sources': data_paths
    }

    output_path = Path(save_dir) / 'patchtst_model.pt'
    torch.save(deployment_package, output_path)
    logger.info(f"完整模型已保存至: {output_path}")

    config_path = Path(save_dir) / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info("訓練完成!")
    logger.info(f"- 最佳模型: {save_dir}/best_model.pt")
    logger.info(f"- 部署模型: {save_dir}/patchtst_model.pt")
    logger.info(f"- 最終 MSE: {final_mse:.6f}")
    logger.info("=" * 60)

    return model, history
