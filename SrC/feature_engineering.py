#!/usr/bin/env python3
"""
高級特徵工程
==============

為神經信號時間序列提取豐富的特徵
"""

import numpy as np
import torch
import torch.nn as nn
from scipy import signal, stats
from typing import Optional, Dict, List
import pywt


class FeatureExtractor:
    """時間序列特徵提取器"""

    @staticmethod
    def extract_statistical_features(x: np.ndarray, axis: int = 0) -> Dict[str, np.ndarray]:
        """
        提取統計特徵

        Args:
            x: 輸入數據 [time, channels] 或 [batch, time, channels]
            axis: 時間維度

        Returns:
            特徵字典
        """
        features = {}

        # 基本統計量
        features['mean'] = np.mean(x, axis=axis)
        features['std'] = np.std(x, axis=axis)
        features['var'] = np.var(x, axis=axis)
        features['min'] = np.min(x, axis=axis)
        features['max'] = np.max(x, axis=axis)
        features['median'] = np.median(x, axis=axis)

        # 高階統計量
        features['skewness'] = stats.skew(x, axis=axis)
        features['kurtosis'] = stats.kurtosis(x, axis=axis)

        # 百分位數
        features['q25'] = np.percentile(x, 25, axis=axis)
        features['q75'] = np.percentile(x, 75, axis=axis)
        features['iqr'] = features['q75'] - features['q25']

        # 變異系數
        features['cv'] = features['std'] / (features['mean'] + 1e-8)

        # 峰值相關
        features['peak_to_peak'] = features['max'] - features['min']
        features['rms'] = np.sqrt(np.mean(x**2, axis=axis))

        return features

    @staticmethod
    def extract_temporal_features(x: np.ndarray) -> Dict[str, np.ndarray]:
        """
        提取時域特徵

        Args:
            x: 輸入數據 [time, channels]

        Returns:
            特徵字典
        """
        features = {}

        # 自相關
        for lag in [1, 5, 10, 20]:
            if x.shape[0] > lag:
                autocorr = []
                for i in range(x.shape[1]):
                    corr = np.corrcoef(x[:-lag, i], x[lag:, i])[0, 1]
                    autocorr.append(corr if not np.isnan(corr) else 0.0)
                features[f'autocorr_lag{lag}'] = np.array(autocorr)

        # 一階差分統計
        diff1 = np.diff(x, axis=0)
        features['diff1_mean'] = np.mean(diff1, axis=0)
        features['diff1_std'] = np.std(diff1, axis=0)

        # 二階差分統計
        diff2 = np.diff(diff1, axis=0)
        features['diff2_mean'] = np.mean(diff2, axis=0)
        features['diff2_std'] = np.std(diff2, axis=0)

        # 過零率
        zero_crossings = np.sum(np.diff(np.sign(x - np.mean(x, axis=0)), axis=0) != 0, axis=0)
        features['zero_crossing_rate'] = zero_crossings / x.shape[0]

        # 能量
        features['energy'] = np.sum(x**2, axis=0)

        # 平均絕對變化
        features['mean_abs_change'] = np.mean(np.abs(diff1), axis=0)

        return features

    @staticmethod
    def extract_frequency_features(x: np.ndarray, fs: float = 100.0) -> Dict[str, np.ndarray]:
        """
        提取頻域特徵

        Args:
            x: 輸入數據 [time, channels]
            fs: 採樣率 (Hz)

        Returns:
            特徵字典
        """
        features = {}

        # FFT 分析
        fft_vals = np.fft.rfft(x, axis=0)
        fft_freq = np.fft.rfftfreq(x.shape[0], 1/fs)
        power_spectrum = np.abs(fft_vals) ** 2

        # 主頻率
        peak_freq_idx = np.argmax(power_spectrum, axis=0)
        features['dominant_freq'] = fft_freq[peak_freq_idx]

        # 頻譜能量
        features['spectral_energy'] = np.sum(power_spectrum, axis=0)

        # 頻譜重心
        features['spectral_centroid'] = np.sum(
            fft_freq[:, np.newaxis] * power_spectrum, axis=0
        ) / (features['spectral_energy'] + 1e-8)

        # 頻譜熵
        power_norm = power_spectrum / (np.sum(power_spectrum, axis=0, keepdims=True) + 1e-8)
        features['spectral_entropy'] = -np.sum(
            power_norm * np.log2(power_norm + 1e-8), axis=0
        )

        # 頻帶能量比例
        # Delta (0.5-4 Hz), Theta (4-8 Hz), Alpha (8-13 Hz), Beta (13-30 Hz), Gamma (30-100 Hz)
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 100)
        }

        for band_name, (low, high) in bands.items():
            band_mask = (fft_freq >= low) & (fft_freq < high)
            band_power = np.sum(power_spectrum[band_mask], axis=0)
            features[f'power_{band_name}'] = band_power
            features[f'power_ratio_{band_name}'] = band_power / (features['spectral_energy'] + 1e-8)

        return features

    @staticmethod
    def extract_wavelet_features(x: np.ndarray, wavelet: str = 'db4', level: int = 3) -> Dict[str, np.ndarray]:
        """
        提取小波特徵

        Args:
            x: 輸入數據 [time, channels]
            wavelet: 小波類型
            level: 分解層數

        Returns:
            特徵字典
        """
        features = {}

        for i in range(x.shape[1]):
            coeffs = pywt.wavedec(x[:, i], wavelet, level=level)

            for j, coeff in enumerate(coeffs):
                prefix = f'wavelet_level{j}_ch{i}'
                features[f'{prefix}_mean'] = np.mean(coeff) if i == 0 else np.append(
                    features.get(f'wavelet_level{j}_mean', np.array([])),
                    np.mean(coeff)
                )
                features[f'{prefix}_std'] = np.std(coeff) if i == 0 else np.append(
                    features.get(f'wavelet_level{j}_std', np.array([])),
                    np.std(coeff)
                )
                features[f'{prefix}_energy'] = np.sum(coeff**2) if i == 0 else np.append(
                    features.get(f'wavelet_level{j}_energy', np.array([])),
                    np.sum(coeff**2)
                )

        return features

    @staticmethod
    def extract_complexity_features(x: np.ndarray) -> Dict[str, np.ndarray]:
        """
        提取複雜度特徵

        Args:
            x: 輸入數據 [time, channels]

        Returns:
            特徵字典
        """
        features = {}

        # 近似熵
        def approx_entropy(signal, m=2, r=None):
            if r is None:
                r = 0.2 * np.std(signal)

            def _maxdist(x_i, x_j):
                return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

            def _phi(m):
                x = [[signal[j] for j in range(i, i + m - 1 + 1)] for i in range(len(signal) - m + 1)]
                C = [len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (len(signal) - m + 1.0) for x_i in x]
                return (len(signal) - m + 1.0) ** (-1) * sum(np.log(C))

            return abs(_phi(m) - _phi(m + 1))

        approx_ent = []
        for i in range(x.shape[1]):
            try:
                ent = approx_entropy(x[:, i])
                approx_ent.append(ent if not np.isnan(ent) else 0.0)
            except:
                approx_ent.append(0.0)

        features['approx_entropy'] = np.array(approx_ent)

        # 樣本熵（簡化版）
        features['sample_entropy'] = features['approx_entropy']  # 使用近似熵作為替代

        # Hurst 指數（長期記憶）
        def hurst_exponent(signal):
            """計算 Hurst 指數"""
            lags = range(2, min(100, len(signal)//2))
            tau = [np.std(np.subtract(signal[lag:], signal[:-lag])) for lag in lags]

            if len(tau) < 2:
                return 0.5

            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0] * 0.5

        hurst = []
        for i in range(x.shape[1]):
            try:
                h = hurst_exponent(x[:, i])
                hurst.append(h if not np.isnan(h) else 0.5)
            except:
                hurst.append(0.5)

        features['hurst_exponent'] = np.array(hurst)

        return features


class FeatureEnhancedModel(nn.Module):
    """
    特徵增強模型包裝器

    將提取的特徵與原始序列結合
    """

    def __init__(
        self,
        base_model: nn.Module,
        use_statistical: bool = True,
        use_frequency: bool = True,
        feature_dim: int = 64
    ):
        super().__init__()
        self.base_model = base_model
        self.use_statistical = use_statistical
        self.use_frequency = use_frequency

        # 特徵融合層
        self.feature_extractor = FeatureExtractor()

        # 特徵編碼器
        n_features = 0
        if use_statistical:
            n_features += 15  # 統計特徵數量
        if use_frequency:
            n_features += 10  # 頻域特徵數量

        if n_features > 0:
            self.feature_encoder = nn.Sequential(
                nn.Linear(n_features, feature_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(feature_dim, feature_dim)
            )

            # 注意力融合
            self.attention = nn.Sequential(
                nn.Linear(feature_dim, 1),
                nn.Sigmoid()
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 基礎模型預測
        base_pred = self.base_model(x)

        # 提取額外特徵
        if self.use_statistical or self.use_frequency:
            x_np = x.cpu().numpy()

            feature_list = []

            if self.use_statistical:
                stat_features = self.feature_extractor.extract_statistical_features(x_np, axis=1)
                # 選擇主要特徵
                selected = ['mean', 'std', 'var', 'min', 'max', 'median',
                           'skewness', 'kurtosis', 'q25', 'q75', 'iqr', 'cv',
                           'peak_to_peak', 'rms', 'rms']
                for key in selected[:15]:
                    if key in stat_features:
                        feature_list.append(stat_features[key])

            if self.use_frequency:
                freq_features = self.feature_extractor.extract_frequency_features(x_np[0])
                # 選擇主要頻域特徵
                selected = ['dominant_freq', 'spectral_energy', 'spectral_centroid',
                           'spectral_entropy', 'power_delta', 'power_theta',
                           'power_alpha', 'power_beta', 'power_gamma', 'power_ratio_alpha']
                for key in selected:
                    if key in freq_features:
                        val = freq_features[key]
                        if val.ndim == 0:
                            val = np.array([val])
                        feature_list.append(val)

            if len(feature_list) > 0:
                # 拼接特徵
                features = np.concatenate([f.flatten()[:x.shape[0]] for f in feature_list])
                features = torch.FloatTensor(features).to(x.device)

                # 調整維度
                if features.shape[0] < self.feature_encoder[0].in_features:
                    features = torch.cat([
                        features,
                        torch.zeros(self.feature_encoder[0].in_features - features.shape[0]).to(x.device)
                    ])
                elif features.shape[0] > self.feature_encoder[0].in_features:
                    features = features[:self.feature_encoder[0].in_features]

                # 編碼特徵
                encoded_features = self.feature_encoder(features)

                # 注意力權重
                attention_weight = self.attention(encoded_features)

                # 特徵調製（簡化版 - 這裡可以更複雜）
                # base_pred: [B, H, N]
                # 可以將特徵作為偏置或縮放因子
                base_pred = base_pred * (1 + 0.1 * attention_weight.item())

        return base_pred


class MultiScaleFeatureExtractor(nn.Module):
    """
    多尺度特徵提取器

    在不同時間尺度上提取特徵
    """

    def __init__(
        self,
        input_size: int,
        scales: List[int] = [1, 2, 4, 8],
        channels: int = 32
    ):
        super().__init__()
        self.scales = scales

        # 多尺度卷積
        self.convs = nn.ModuleList([
            nn.Conv1d(1, channels, kernel_size=scale*2+1, padding=scale)
            for scale in scales
        ])

        # 特徵融合
        self.fusion = nn.Conv1d(channels * len(scales), channels, kernel_size=1)

        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, n_vars]

        Returns:
            [batch, seq_len, channels]
        """
        B, T, N = x.shape

        # 處理每個變量
        outputs = []

        for i in range(N):
            x_var = x[:, :, i:i+1].permute(0, 2, 1)  # [B, 1, T]

            scale_features = []
            for conv in self.convs:
                feat = self.activation(conv(x_var))  # [B, channels, T]
                scale_features.append(feat)

            # 拼接多尺度特徵
            multi_scale = torch.cat(scale_features, dim=1)  # [B, channels*scales, T]

            # 融合
            fused = self.fusion(multi_scale)  # [B, channels, T]
            outputs.append(fused)

        # 合併所有變量
        output = torch.stack(outputs, dim=-1).mean(dim=-1)  # [B, channels, T]
        output = output.permute(0, 2, 1)  # [B, T, channels]

        return output
