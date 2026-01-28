#!/usr/bin/env python3
"""
數據增強模塊 - 針對神經信號時間序列
================================================

為神經預測比賽設計的專業數據增強策略
"""

import numpy as np
from typing import Optional, Tuple
import torch


class TimeSeriesAugmenter:
    """時間序列數據增強器"""

    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

    def jitter(self, x: np.ndarray, sigma: float = 0.03) -> np.ndarray:
        """
        添加高斯噪聲 - 模擬測量噪聲

        Args:
            x: 輸入數據 (time, channels)
            sigma: 噪聲標準差（相對於數據標準差）
        """
        noise = np.random.normal(0, sigma * np.std(x), x.shape)
        return x + noise

    def scaling(self, x: np.ndarray, sigma: float = 0.1) -> np.ndarray:
        """
        隨機縮放 - 模擬放電率變化

        Args:
            x: 輸入數據
            sigma: 縮放因子標準差
        """
        factor = np.random.normal(1.0, sigma, (1, x.shape[1]))
        return x * factor

    def time_warp(self, x: np.ndarray, sigma: float = 0.2, knot: int = 4) -> np.ndarray:
        """
        時間扭曲 - 非線性時間伸縮

        Args:
            x: 輸入數據 (time, channels)
            sigma: 扭曲強度
            knot: 扭曲節點數
        """
        from scipy.interpolate import CubicSpline

        orig_steps = np.arange(x.shape[0])

        random_warps = np.random.normal(loc=1.0, scale=sigma, size=(knot+2, x.shape[1]))
        warp_steps = np.linspace(0, x.shape[0]-1, num=knot+2)

        ret = np.zeros_like(x)
        for i in range(x.shape[1]):
            time_warp_func = CubicSpline(warp_steps, warp_steps * random_warps[:, i])
            warped_steps = time_warp_func(orig_steps)
            warped_steps = np.clip(warped_steps, 0, x.shape[0]-1)

            # 插值到扭曲後的時間點
            ret[:, i] = np.interp(orig_steps, warped_steps, x[:, i])

        return ret

    def magnitude_warp(self, x: np.ndarray, sigma: float = 0.2, knot: int = 4) -> np.ndarray:
        """
        幅度扭曲 - 平滑的幅度變化

        Args:
            x: 輸入數據
            sigma: 扭曲強度
            knot: 扭曲節點數
        """
        from scipy.interpolate import CubicSpline

        orig_steps = np.arange(x.shape[0])
        random_warps = np.random.normal(loc=1.0, scale=sigma, size=(knot+2, x.shape[1]))
        warp_steps = np.linspace(0, x.shape[0]-1, num=knot+2)

        ret = np.zeros_like(x)
        for i in range(x.shape[1]):
            magnitude_warp_func = CubicSpline(warp_steps, random_warps[:, i])
            ret[:, i] = x[:, i] * magnitude_warp_func(orig_steps)

        return ret

    def window_slice(self, x: np.ndarray, reduce_ratio: float = 0.9) -> np.ndarray:
        """
        窗口切片 - 隨機選擇連續子序列

        Args:
            x: 輸入數據
            reduce_ratio: 保留比例
        """
        target_len = int(x.shape[0] * reduce_ratio)
        if target_len >= x.shape[0]:
            return x

        start = np.random.randint(0, x.shape[0] - target_len)
        return x[start:start+target_len, :]

    def window_warp(self, x: np.ndarray, window_ratio: float = 0.1, scales: Tuple[float, float] = (0.5, 2.0)) -> np.ndarray:
        """
        窗口扭曲 - 局部時間伸縮

        Args:
            x: 輸入數據
            window_ratio: 窗口大小比例
            scales: 縮放範圍 (min, max)
        """
        from scipy.interpolate import interp1d

        warp_size = int(x.shape[0] * window_ratio)
        window_start = np.random.randint(0, x.shape[0] - warp_size)
        window_end = window_start + warp_size

        scale = np.random.uniform(scales[0], scales[1])

        warped = x.copy()
        for i in range(x.shape[1]):
            # 提取窗口並縮放
            window = x[window_start:window_end, i]
            orig_steps = np.arange(warp_size)

            # 重採樣
            new_size = int(warp_size * scale)
            new_steps = np.linspace(0, warp_size-1, new_size)

            if new_size > 0:
                interp_func = interp1d(orig_steps, window, kind='linear', fill_value='extrapolate')
                warped_window = interp_func(new_steps)

                # 插入回原序列
                if new_size <= warp_size:
                    warped[window_start:window_start+new_size, i] = warped_window
                    warped[window_start+new_size:window_end, i] = x[window_end-warp_size+new_size:window_end, i]
                else:
                    # 如果變長了，需要截斷或壓縮
                    warped[window_start:window_end, i] = warped_window[:warp_size]

        return warped

    def channel_shuffle(self, x: np.ndarray) -> np.ndarray:
        """
        通道洗牌 - 隨機排列神經元通道

        適用於假設神經元獨立的場景
        """
        indices = np.random.permutation(x.shape[1])
        return x[:, indices]

    def dropout_channels(self, x: np.ndarray, dropout_ratio: float = 0.1) -> np.ndarray:
        """
        通道 Dropout - 隨機置零部分通道

        模擬神經元失活或記錄失敗
        """
        mask = np.random.binomial(1, 1-dropout_ratio, x.shape[1])
        return x * mask[np.newaxis, :]

    def mixup(self, x1: np.ndarray, x2: np.ndarray, alpha: float = 0.2) -> np.ndarray:
        """
        MixUp 增強 - 線性混合兩個樣本

        Args:
            x1, x2: 兩個輸入樣本
            alpha: Beta 分佈參數
        """
        lam = np.random.beta(alpha, alpha)
        return lam * x1 + (1 - lam) * x2

    def cutmix_time(self, x1: np.ndarray, x2: np.ndarray, alpha: float = 0.2) -> np.ndarray:
        """
        CutMix (時間維度) - 切割並混合時間片段

        Args:
            x1, x2: 兩個輸入樣本
            alpha: Beta 分佈參數
        """
        lam = np.random.beta(alpha, alpha)
        cut_len = int(x1.shape[0] * (1 - lam))
        cut_start = np.random.randint(0, x1.shape[0] - cut_len + 1)

        x_mixed = x1.copy()
        x_mixed[cut_start:cut_start+cut_len] = x2[cut_start:cut_start+cut_len]

        return x_mixed

    def frequency_masking(self, x: np.ndarray, mask_ratio: float = 0.1) -> np.ndarray:
        """
        頻域遮罩 - 在頻域隨機遮罩部分頻率

        Args:
            x: 輸入數據
            mask_ratio: 遮罩比例
        """
        x_fft = np.fft.rfft(x, axis=0)
        n_freqs = x_fft.shape[0]
        n_mask = int(n_freqs * mask_ratio)

        mask_start = np.random.randint(0, n_freqs - n_mask + 1)
        x_fft[mask_start:mask_start+n_mask] = 0

        return np.fft.irfft(x_fft, n=x.shape[0], axis=0).real

    def add_trend(self, x: np.ndarray, trend_strength: float = 0.1) -> np.ndarray:
        """
        添加隨機趨勢 - 模擬非平穩性

        Args:
            x: 輸入數據
            trend_strength: 趨勢強度（相對於數據標準差）
        """
        t = np.linspace(0, 1, x.shape[0])
        trends = np.random.randn(x.shape[1]) * np.std(x, axis=0) * trend_strength
        trend_component = t[:, np.newaxis] * trends[np.newaxis, :]
        return x + trend_component

    def random_augment(self, x: np.ndarray, n_ops: int = 2, exclude: Optional[list] = None) -> np.ndarray:
        """
        隨機組合增強 - 隨機選擇並應用多個增強操作

        Args:
            x: 輸入數據
            n_ops: 應用的操作數量
            exclude: 排除的操作列表
        """
        ops = [
            ('jitter', lambda x: self.jitter(x, sigma=0.03)),
            ('scaling', lambda x: self.scaling(x, sigma=0.1)),
            ('magnitude_warp', lambda x: self.magnitude_warp(x, sigma=0.2)),
            ('time_warp', lambda x: self.time_warp(x, sigma=0.2)),
            ('dropout_channels', lambda x: self.dropout_channels(x, dropout_ratio=0.1)),
        ]

        if exclude:
            ops = [(name, func) for name, func in ops if name not in exclude]

        selected_ops = np.random.choice(len(ops), size=min(n_ops, len(ops)), replace=False)

        result = x.copy()
        for idx in selected_ops:
            name, func = ops[idx]
            try:
                result = func(result)
            except Exception as e:
                # 如果某個增強失敗，跳過
                print(f"Augmentation {name} failed: {e}")
                continue

        return result


class AugmentedDataset(torch.utils.data.Dataset):
    """
    支援即時增強的數據集
    """

    def __init__(
        self,
        base_dataset: torch.utils.data.Dataset,
        augmenter: TimeSeriesAugmenter,
        augment_prob: float = 0.5,
        n_augment_ops: int = 2
    ):
        self.base_dataset = base_dataset
        self.augmenter = augmenter
        self.augment_prob = augment_prob
        self.n_augment_ops = n_augment_ops

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = self.base_dataset[idx]

        # 以一定概率應用增強
        if np.random.rand() < self.augment_prob:
            x_np = x.numpy()
            x_aug = self.augmenter.random_augment(x_np, n_ops=self.n_augment_ops)
            x = torch.FloatTensor(x_aug)

        return x, y
