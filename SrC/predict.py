#!/usr/bin/env python3
"""
Codabench 提交腳本 - 預測接口
==================================

用於生成比賽提交的預測結果
"""

import sys
import os
import argparse
from pathlib import Path
import numpy as np
import torch

# 添加 SrC 到路徑
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'SrC'))

from models_PatchTST import PatchTST, PatchTSTConfig
from Training_PatchTST import PatchTSTPredictor, gaussian_smooth, sqrt_transform


class CompetitionPredictor:
    """
    比賽預測器 - 統一接口

    支援單模型和集成模型
    """

    def __init__(self, model_paths: list, device: str = 'auto'):
        """
        Args:
            model_paths: 模型文件路徑列表
            device: 設備 ('cuda', 'cpu', 'auto')
        """
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        print(f"使用設備: {self.device}")

        # 加載所有模型
        self.predictors = []
        for i, path in enumerate(model_paths):
            print(f"加載模型 {i+1}/{len(model_paths)}: {path}")
            predictor = PatchTSTPredictor.load(path, device=self.device)
            self.predictors.append(predictor)

        print(f"成功加載 {len(self.predictors)} 個模型")

    def predict(
        self,
        input_data: np.ndarray,
        ensemble_method: str = 'average',
        preprocess: bool = True
    ) -> np.ndarray:
        """
        生成預測

        Args:
            input_data: 輸入數據 [seq_len, n_channels] 或 [batch, seq_len, n_channels]
            ensemble_method: 集成方法 ('average', 'median', 'weighted')
            preprocess: 是否預處理

        Returns:
            預測結果 [horizon, n_channels] 或 [batch, horizon, n_channels]
        """
        if input_data.ndim == 2:
            input_data = input_data[np.newaxis, ...]
            squeeze_output = True
        else:
            squeeze_output = False

        # 每個模型預測
        predictions = []
        for predictor in self.predictors:
            pred = predictor.predict(input_data, preprocess=preprocess)
            predictions.append(pred)

        # 集成
        predictions = np.stack(predictions, axis=0)  # [n_models, batch, horizon, n_channels]

        if ensemble_method == 'average':
            final_pred = np.mean(predictions, axis=0)
        elif ensemble_method == 'median':
            final_pred = np.median(predictions, axis=0)
        elif ensemble_method == 'weighted':
            # 簡化版加權：基於驗證性能
            # 實際應用中應該從訓練時保存權重
            weights = np.array([1.0] * len(self.predictors))
            weights = weights / weights.sum()
            final_pred = np.sum(predictions * weights[:, None, None, None], axis=0)
        else:
            raise ValueError(f"Unknown ensemble method: {ensemble_method}")

        if squeeze_output:
            final_pred = final_pred.squeeze(0)

        return final_pred

    def predict_batch(
        self,
        input_batch: np.ndarray,
        batch_size: int = 32,
        **kwargs
    ) -> np.ndarray:
        """
        批量預測（大數據集）

        Args:
            input_batch: 輸入批次 [n_samples, seq_len, n_channels]
            batch_size: 批次大小

        Returns:
            預測結果 [n_samples, horizon, n_channels]
        """
        n_samples = input_batch.shape[0]
        predictions = []

        for i in range(0, n_samples, batch_size):
            batch = input_batch[i:i+batch_size]
            pred = self.predict(batch, **kwargs)
            predictions.append(pred)

        return np.concatenate(predictions, axis=0)


def load_test_data(test_file: str) -> np.ndarray:
    """
    加載測試數據

    Args:
        test_file: 測試數據文件路徑 (.npz)

    Returns:
        測試數據數組
    """
    print(f"加載測試數據: {test_file}")

    with np.load(test_file) as npz:
        keys = list(npz.keys())
        print(f"可用 keys: {keys}")

        # 嘗試標準 key
        if 'data' in keys:
            data = npz['data']
        elif 'test_data' in keys:
            data = npz['test_data']
        elif 'X_test' in keys:
            data = npz['X_test']
        elif 'arr_0' in keys:
            data = npz['arr_0']
        else:
            data = npz[keys[0]]

    print(f"測試數據形狀: {data.shape}")
    return data


def save_predictions(predictions: np.ndarray, output_file: str):
    """
    保存預測結果

    Args:
        predictions: 預測數組
        output_file: 輸出文件路徑
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 根據文件擴展名選擇保存格式
    if output_path.suffix == '.npz':
        np.savez_compressed(output_file, predictions=predictions)
    elif output_path.suffix == '.npy':
        np.save(output_file, predictions)
    elif output_path.suffix == '.csv':
        # CSV 格式：展平為 2D
        if predictions.ndim == 3:
            # [n_samples, horizon, n_channels] -> [n_samples, horizon*n_channels]
            predictions_2d = predictions.reshape(predictions.shape[0], -1)
        else:
            predictions_2d = predictions
        np.savetxt(output_file, predictions_2d, delimiter=',')
    else:
        raise ValueError(f"Unsupported output format: {output_path.suffix}")

    print(f"預測結果已保存: {output_file}")
    print(f"輸出形狀: {predictions.shape}")


def main():
    parser = argparse.ArgumentParser(description='Neural Forecasting Competition - Prediction')

    parser.add_argument(
        '--test-data',
        type=str,
        required=True,
        help='測試數據文件路徑 (.npz)'
    )

    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        required=True,
        help='模型文件路徑（可以多個用於集成）'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='predictions.npz',
        help='輸出文件路徑'
    )

    parser.add_argument(
        '--ensemble',
        type=str,
        default='average',
        choices=['average', 'median', 'weighted'],
        help='集成方法'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='批次大小'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='計算設備'
    )

    parser.add_argument(
        '--no-preprocess',
        action='store_true',
        help='禁用預處理'
    )

    args = parser.parse_args()

    # 加載測試數據
    test_data = load_test_data(args.test_data)

    # 創建預測器
    predictor = CompetitionPredictor(
        model_paths=args.models,
        device=args.device
    )

    # 生成預測
    print("\n開始預測...")
    predictions = predictor.predict_batch(
        test_data,
        batch_size=args.batch_size,
        ensemble_method=args.ensemble,
        preprocess=not args.no_preprocess
    )

    # 保存結果
    save_predictions(predictions, args.output)

    print("\n預測完成！")
    print(f"預測形狀: {predictions.shape}")
    print(f"預測範圍: [{predictions.min():.4f}, {predictions.max():.4f}]")
    print(f"預測均值: {predictions.mean():.4f}")
    print(f"預測標準差: {predictions.std():.4f}")


if __name__ == '__main__':
    main()
