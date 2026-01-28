#!/usr/bin/env python3
"""
超參數調優和交叉驗證
====================

實現時間序列交叉驗證和超參數搜索
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from typing import Dict, List, Tuple, Any, Callable, Optional
from pathlib import Path
import json
import itertools
from tqdm import tqdm
import copy


class TimeSeriesCV:
    """
    時間序列交叉驗證

    使用滑動窗口方式劃分訓練/驗證集
    確保不會有未來數據洩漏到訓練集
    """

    def __init__(
        self,
        n_splits: int = 5,
        test_size: Optional[int] = None,
        gap: int = 0
    ):
        """
        Args:
            n_splits: 交叉驗證折數
            test_size: 測試集大小（None 則自動計算）
            gap: 訓練集和測試集之間的間隔
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap

    def split(self, data: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        生成交叉驗證索引

        Args:
            data: 時間序列數據 [n_samples, ...]

        Returns:
            [(train_indices, val_indices), ...]
        """
        n_samples = len(data)

        if self.test_size is None:
            test_size = n_samples // (self.n_splits + 1)
        else:
            test_size = self.test_size

        splits = []

        for i in range(self.n_splits):
            # 計算測試集範圍
            test_end = n_samples - i * test_size
            test_start = test_end - test_size

            if test_start <= 0:
                break

            # 訓練集：從開始到測試集之前（考慮 gap）
            train_end = test_start - self.gap
            train_indices = np.arange(0, train_end)
            val_indices = np.arange(test_start, test_end)

            if len(train_indices) > 0 and len(val_indices) > 0:
                splits.append((train_indices, val_indices))

        return splits[::-1]  # 逆序，從早期到晚期

    def split_dataset(
        self,
        dataset: Dataset,
        total_samples: int
    ) -> List[Tuple[Subset, Subset]]:
        """
        基於 PyTorch Dataset 進行劃分

        Args:
            dataset: PyTorch Dataset
            total_samples: 總樣本數

        Returns:
            [(train_subset, val_subset), ...]
        """
        splits = self.split(np.arange(total_samples))

        dataset_splits = []
        for train_idx, val_idx in splits:
            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)
            dataset_splits.append((train_subset, val_subset))

        return dataset_splits


class HyperparameterSearcher:
    """
    超參數搜索器

    支援網格搜索和隨機搜索
    """

    def __init__(
        self,
        model_factory: Callable,
        trainer_factory: Callable,
        param_grid: Dict[str, List[Any]],
        cv: Optional[TimeSeriesCV] = None,
        scoring: str = 'mse',
        n_jobs: int = 1
    ):
        """
        Args:
            model_factory: 模型創建函數 (params) -> model
            trainer_factory: 訓練器創建函數 (model, params) -> trainer
            param_grid: 參數網格 {'param_name': [val1, val2, ...]}
            cv: 交叉驗證對象
            scoring: 評分指標
            n_jobs: 並行任務數（暫未實現）
        """
        self.model_factory = model_factory
        self.trainer_factory = trainer_factory
        self.param_grid = param_grid
        self.cv = cv if cv is not None else TimeSeriesCV(n_splits=3)
        self.scoring = scoring

        self.results = []
        self.best_params = None
        self.best_score = float('inf') if scoring in ['mse', 'mae'] else float('-inf')

    def grid_search(
        self,
        train_data: np.ndarray,
        dataset_class: Callable,
        base_config: Dict,
        max_trials: Optional[int] = None
    ) -> Dict:
        """
        網格搜索

        Args:
            train_data: 訓練數據
            dataset_class: Dataset 類
            base_config: 基礎配置
            max_trials: 最大嘗試次數

        Returns:
            搜索結果字典
        """
        # 生成所有參數組合
        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())
        param_combinations = list(itertools.product(*param_values))

        if max_trials is not None:
            param_combinations = param_combinations[:max_trials]

        print(f"網格搜索 - 總共 {len(param_combinations)} 種組合")

        for i, param_values in enumerate(tqdm(param_combinations, desc="Grid Search")):
            params = dict(zip(param_names, param_values))

            # 合併參數
            config = {**base_config, **params}

            # 交叉驗證評估
            cv_scores = self._cross_validate(train_data, dataset_class, config)

            mean_score = np.mean(cv_scores)
            std_score = np.std(cv_scores)

            self.results.append({
                'params': params,
                'cv_scores': cv_scores,
                'mean_score': mean_score,
                'std_score': std_score
            })

            # 更新最佳參數
            if mean_score < self.best_score:  # 假設分數越小越好
                self.best_score = mean_score
                self.best_params = params

            print(f"\n參數: {params}")
            print(f"CV 分數: {mean_score:.6f} ± {std_score:.6f}")

        print(f"\n最佳參數: {self.best_params}")
        print(f"最佳分數: {self.best_score:.6f}")

        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'all_results': self.results
        }

    def random_search(
        self,
        train_data: np.ndarray,
        dataset_class: Callable,
        base_config: Dict,
        n_iter: int = 20
    ) -> Dict:
        """
        隨機搜索

        Args:
            train_data: 訓練數據
            dataset_class: Dataset 類
            base_config: 基礎配置
            n_iter: 隨機嘗試次數

        Returns:
            搜索結果字典
        """
        print(f"隨機搜索 - 嘗試 {n_iter} 次")

        param_names = list(self.param_grid.keys())

        for i in tqdm(range(n_iter), desc="Random Search"):
            # 隨機選擇參數
            params = {}
            for name in param_names:
                params[name] = np.random.choice(self.param_grid[name])

            # 合併參數
            config = {**base_config, **params}

            # 交叉驗證評估
            cv_scores = self._cross_validate(train_data, dataset_class, config)

            mean_score = np.mean(cv_scores)
            std_score = np.std(cv_scores)

            self.results.append({
                'params': params,
                'cv_scores': cv_scores,
                'mean_score': mean_score,
                'std_score': std_score
            })

            # 更新最佳參數
            if mean_score < self.best_score:
                self.best_score = mean_score
                self.best_params = params

            if (i + 1) % 5 == 0:
                print(f"\n當前最佳分數: {self.best_score:.6f}")
                print(f"當前最佳參數: {self.best_params}")

        print(f"\n最終最佳參數: {self.best_params}")
        print(f"最終最佳分數: {self.best_score:.6f}")

        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'all_results': self.results
        }

    def _cross_validate(
        self,
        train_data: np.ndarray,
        dataset_class: Callable,
        config: Dict
    ) -> List[float]:
        """
        執行交叉驗證

        Returns:
            每折的驗證分數列表
        """
        # 創建完整數據集
        full_dataset = dataset_class(
            train_data,
            input_size=config.get('input_size', 96),
            horizon=config.get('horizon', 24),
            transform=True
        )

        # 交叉驗證劃分
        cv_splits = self.cv.split_dataset(full_dataset, len(full_dataset))

        scores = []

        for fold, (train_subset, val_subset) in enumerate(cv_splits):
            # 創建數據加載器
            train_loader = DataLoader(
                train_subset,
                batch_size=config.get('batch_size', 32),
                shuffle=True,
                num_workers=0
            )

            val_loader = DataLoader(
                val_subset,
                batch_size=config.get('batch_size', 32),
                shuffle=False,
                num_workers=0
            )

            # 創建模型和訓練器
            model = self.model_factory(config)
            trainer = self.trainer_factory(model, config)

            # 訓練
            trainer.fit(train_loader, val_loader)

            # 評估
            val_loss, _ = trainer.validate(val_loader)
            scores.append(val_loss)

        return scores

    def save_results(self, save_path: str):
        """保存搜索結果"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        results_dict = {
            'best_params': self.best_params,
            'best_score': float(self.best_score),
            'all_results': [
                {
                    'params': r['params'],
                    'cv_scores': [float(s) for s in r['cv_scores']],
                    'mean_score': float(r['mean_score']),
                    'std_score': float(r['std_score'])
                }
                for r in self.results
            ]
        }

        with open(save_path, 'w') as f:
            json.dump(results_dict, f, indent=2)

        print(f"搜索結果已保存至: {save_path}")


class BayesianOptimizer:
    """
    貝葉斯優化（簡化版）

    使用高斯過程進行更智能的超參數搜索
    """

    def __init__(
        self,
        model_factory: Callable,
        trainer_factory: Callable,
        param_bounds: Dict[str, Tuple[float, float]],
        cv: Optional[TimeSeriesCV] = None
    ):
        """
        Args:
            model_factory: 模型創建函數
            trainer_factory: 訓練器創建函數
            param_bounds: 參數範圍 {'param': (min, max)}
            cv: 交叉驗證對象
        """
        self.model_factory = model_factory
        self.trainer_factory = trainer_factory
        self.param_bounds = param_bounds
        self.cv = cv if cv is not None else TimeSeriesCV(n_splits=3)

        self.history = []
        self.best_params = None
        self.best_score = float('inf')

    def optimize(
        self,
        train_data: np.ndarray,
        dataset_class: Callable,
        base_config: Dict,
        n_iter: int = 30,
        n_random_starts: int = 10
    ) -> Dict:
        """
        執行貝葉斯優化

        Args:
            train_data: 訓練數據
            dataset_class: Dataset 類
            base_config: 基礎配置
            n_iter: 總迭代次數
            n_random_starts: 隨機初始化次數

        Returns:
            優化結果
        """
        print(f"貝葉斯優化 - {n_iter} 次迭代")

        param_names = list(self.param_bounds.keys())

        # 隨機初始化
        for i in tqdm(range(n_random_starts), desc="Random Initialization"):
            params = self._sample_random_params()
            score = self._evaluate_params(params, train_data, dataset_class, base_config)

            self.history.append({'params': params, 'score': score})

            if score < self.best_score:
                self.best_score = score
                self.best_params = params

        # 貝葉斯優化迭代
        for i in tqdm(range(n_random_starts, n_iter), desc="Bayesian Optimization"):
            # 簡化版：基於歷史最佳附近隨機採樣
            # 真正的貝葉斯優化需要使用 GPyOpt 或 Optuna
            params = self._sample_near_best()
            score = self._evaluate_params(params, train_data, dataset_class, base_config)

            self.history.append({'params': params, 'score': score})

            if score < self.best_score:
                self.best_score = score
                self.best_params = params

            if (i + 1) % 5 == 0:
                print(f"\n當前最佳分數: {self.best_score:.6f}")

        print(f"\n最終最佳參數: {self.best_params}")
        print(f"最終最佳分數: {self.best_score:.6f}")

        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'history': self.history
        }

    def _sample_random_params(self) -> Dict:
        """隨機採樣參數"""
        params = {}
        for name, (low, high) in self.param_bounds.items():
            if isinstance(low, int) and isinstance(high, int):
                params[name] = np.random.randint(low, high + 1)
            else:
                params[name] = np.random.uniform(low, high)
        return params

    def _sample_near_best(self, std_ratio: float = 0.1) -> Dict:
        """在最佳參數附近採樣"""
        if self.best_params is None:
            return self._sample_random_params()

        params = {}
        for name, (low, high) in self.param_bounds.items():
            best_val = self.best_params[name]
            std = (high - low) * std_ratio

            if isinstance(low, int) and isinstance(high, int):
                new_val = int(np.random.normal(best_val, std))
                params[name] = np.clip(new_val, low, high)
            else:
                new_val = np.random.normal(best_val, std)
                params[name] = np.clip(new_val, low, high)

        return params

    def _evaluate_params(
        self,
        params: Dict,
        train_data: np.ndarray,
        dataset_class: Callable,
        base_config: Dict
    ) -> float:
        """評估參數組合"""
        config = {**base_config, **params}

        # 簡化：只用一次訓練-驗證劃分
        from Training_PatchTST import NeuralDataset

        split_idx = int(len(train_data) * 0.8)
        train_subset = train_data[:split_idx]
        val_subset = train_data[split_idx:]

        train_dataset = NeuralDataset(
            train_subset,
            input_size=config.get('input_size', 96),
            horizon=config.get('horizon', 24),
            transform=True
        )

        val_dataset = NeuralDataset(
            val_subset,
            input_size=config.get('input_size', 96),
            horizon=config.get('horizon', 24),
            transform=True
        )

        train_loader = DataLoader(train_dataset, batch_size=config.get('batch_size', 32), shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.get('batch_size', 32), shuffle=False)

        model = self.model_factory(config)
        trainer = self.trainer_factory(model, config)

        # 減少訓練輪數以加速搜索
        original_epochs = config.get('max_epochs', 50)
        config['max_epochs'] = min(original_epochs, 20)

        trainer.fit(train_loader, val_loader)

        val_loss, _ = trainer.validate(val_loader)
        return val_loss
