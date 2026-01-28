#!/usr/bin/env python3
"""
模型集成框架
=============

實現多種集成策略以提升預測性能
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Optional, Callable
from pathlib import Path
import json


class EnsemblePredictor:
    """
    模型集成預測器

    支援多種集成策略:
    - Simple Average: 簡單平均
    - Weighted Average: 加權平均
    - Stacking: 元學習器堆疊
    - Rank Average: 排名平均
    """

    def __init__(self, models: List[nn.Module], device: str = 'cpu'):
        self.models = models
        self.device = device
        self.weights = None
        self.meta_learner = None

        for model in self.models:
            model.to(device)
            model.eval()

    def predict_average(self, x: torch.Tensor) -> torch.Tensor:
        """簡單平均集成"""
        x = x.to(self.device)
        predictions = []

        with torch.no_grad():
            for model in self.models:
                pred = model(x)
                predictions.append(pred)

        return torch.stack(predictions).mean(dim=0)

    def predict_weighted(self, x: torch.Tensor, weights: Optional[np.ndarray] = None) -> torch.Tensor:
        """加權平均集成"""
        if weights is None:
            if self.weights is None:
                return self.predict_average(x)
            weights = self.weights
        else:
            weights = np.array(weights)
            weights = weights / weights.sum()

        x = x.to(self.device)
        predictions = []

        with torch.no_grad():
            for model in self.models:
                pred = model(x)
                predictions.append(pred)

        weighted_pred = torch.zeros_like(predictions[0])
        for i, pred in enumerate(predictions):
            weighted_pred += pred * weights[i]

        return weighted_pred

    def predict_median(self, x: torch.Tensor) -> torch.Tensor:
        """中位數集成 - 對異常值更穩健"""
        x = x.to(self.device)
        predictions = []

        with torch.no_grad():
            for model in self.models:
                pred = model(x)
                predictions.append(pred)

        return torch.stack(predictions).median(dim=0)[0]

    def compute_optimal_weights(
        self,
        val_loader: torch.utils.data.DataLoader,
        method: str = 'mse'
    ) -> np.ndarray:
        """
        基於驗證集計算最優權重

        Args:
            val_loader: 驗證數據加載器
            method: 'mse' - 基於 MSE, 'inverse_loss' - 反損失加權

        Returns:
            最優權重數組
        """
        print("計算每個模型在驗證集上的性能...")

        # 計算每個模型的驗證損失
        losses = []
        criterion = nn.MSELoss()

        for i, model in enumerate(self.models):
            model.eval()
            total_loss = 0.0
            n_batches = 0

            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    pred = model(x)
                    loss = criterion(pred, y)
                    total_loss += loss.item()
                    n_batches += 1

            avg_loss = total_loss / n_batches
            losses.append(avg_loss)
            print(f"  模型 {i+1}: MSE = {avg_loss:.6f}")

        losses = np.array(losses)

        # 基於反損失計算權重
        if method == 'inverse_loss':
            weights = 1.0 / (losses + 1e-8)
            weights = weights / weights.sum()
        elif method == 'softmax':
            # 使用 softmax（損失越小權重越大）
            weights = np.exp(-losses)
            weights = weights / weights.sum()
        else:
            # 均勻權重
            weights = np.ones(len(losses)) / len(losses)

        self.weights = weights
        print(f"\n最優權重: {weights}")
        return weights

    def train_stacking(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        meta_model: Optional[nn.Module] = None,
        epochs: int = 20
    ):
        """
        訓練 Stacking 元學習器

        Args:
            train_loader: 訓練數據加載器
            val_loader: 驗證數據加載器
            meta_model: 元學習器模型（None 則使用簡單線性模型）
            epochs: 訓練輪數
        """
        print("訓練 Stacking 元學習器...")

        # 生成基礎模型的預測作為元特徵
        def generate_meta_features(loader):
            X_meta_list = []
            y_list = []

            with torch.no_grad():
                for x, y in loader:
                    x = x.to(self.device)
                    preds = []
                    for model in self.models:
                        pred = model(x)
                        preds.append(pred)

                    # 拼接所有模型的預測作為元特徵
                    X_meta = torch.cat(preds, dim=-1)  # [B, H, N*M]
                    X_meta_list.append(X_meta.cpu())
                    y_list.append(y.cpu())

            return torch.cat(X_meta_list, dim=0), torch.cat(y_list, dim=0)

        X_train_meta, y_train = generate_meta_features(train_loader)
        X_val_meta, y_val = generate_meta_features(val_loader)

        # 創建元學習器
        if meta_model is None:
            input_dim = X_train_meta.shape[-1]
            output_dim = y_train.shape[-1]
            meta_model = nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(input_dim // 2, output_dim)
            )

        meta_model = meta_model.to(self.device)
        optimizer = torch.optim.Adam(meta_model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        best_val_loss = float('inf')

        for epoch in range(epochs):
            meta_model.train()

            X_batch = X_train_meta.to(self.device)
            y_batch = y_train.to(self.device)

            pred = meta_model(X_batch)
            loss = criterion(pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 驗證
            meta_model.eval()
            with torch.no_grad():
                X_val_batch = X_val_meta.to(self.device)
                y_val_batch = y_val.to(self.device)
                val_pred = meta_model(X_val_batch)
                val_loss = criterion(val_pred, y_val_batch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss

            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{epochs} - Train Loss: {loss.item():.6f}, Val Loss: {val_loss.item():.6f}")

        self.meta_learner = meta_model
        print(f"Stacking 訓練完成 - 最佳驗證 MSE: {best_val_loss:.6f}")

    def predict_stacking(self, x: torch.Tensor) -> torch.Tensor:
        """使用 Stacking 元學習器預測"""
        if self.meta_learner is None:
            raise ValueError("需要先訓練元學習器！調用 train_stacking()")

        x = x.to(self.device)

        # 生成基礎模型預測
        with torch.no_grad():
            preds = []
            for model in self.models:
                pred = model(x)
                preds.append(pred)

            X_meta = torch.cat(preds, dim=-1)

            # 元學習器預測
            final_pred = self.meta_learner(X_meta)

        return final_pred

    def save(self, save_dir: str):
        """保存集成模型"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # 保存每個模型
        for i, model in enumerate(self.models):
            model_path = save_dir / f"model_{i}.pt"
            torch.save(model.state_dict(), model_path)

        # 保存權重
        if self.weights is not None:
            np.save(save_dir / "weights.npy", self.weights)

        # 保存元學習器
        if self.meta_learner is not None:
            torch.save(self.meta_learner.state_dict(), save_dir / "meta_learner.pt")

        print(f"集成模型已保存至: {save_dir}")

    @classmethod
    def load(cls, load_dir: str, model_classes: List[Callable], device: str = 'cpu'):
        """加載集成模型"""
        load_dir = Path(load_dir)

        models = []
        for i, model_class in enumerate(model_classes):
            model_path = load_dir / f"model_{i}.pt"
            model = model_class()
            model.load_state_dict(torch.load(model_path, map_location=device))
            models.append(model)

        ensemble = cls(models, device)

        # 加載權重
        weights_path = load_dir / "weights.npy"
        if weights_path.exists():
            ensemble.weights = np.load(weights_path)

        # 加載元學習器
        meta_path = load_dir / "meta_learner.pt"
        if meta_path.exists():
            # TODO: 需要知道元學習器的架構
            pass

        return ensemble


class AdaptiveEnsemble:
    """
    自適應集成 - 根據輸入動態調整權重
    """

    def __init__(self, models: List[nn.Module], device: str = 'cpu'):
        self.models = models
        self.device = device

        # 權重網絡 - 輸入序列 -> 模型權重
        self.weight_network = None

    def train_weight_network(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        input_size: int,
        epochs: int = 30
    ):
        """
        訓練自適應權重網絡

        根據輸入序列特徵動態生成模型權重
        """
        print("訓練自適應權重網絡...")

        # 創建權重網絡
        n_models = len(self.models)
        self.weight_network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_models),
            nn.Softmax(dim=-1)
        ).to(self.device)

        optimizer = torch.optim.Adam(self.weight_network.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        best_val_loss = float('inf')

        for epoch in range(epochs):
            self.weight_network.train()
            train_loss = 0.0
            n_batches = 0

            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)

                # 計算動態權重
                x_features = x.mean(dim=-1)  # 聚合通道特徵
                weights = self.weight_network(x_features)  # [B, n_models]

                # 獲取每個模型的預測
                with torch.no_grad():
                    preds = []
                    for model in self.models:
                        pred = model(x)
                        preds.append(pred)
                    preds = torch.stack(preds, dim=0)  # [n_models, B, H, N]

                # 加權組合
                weights_expanded = weights.T.unsqueeze(-1).unsqueeze(-1)  # [n_models, B, 1, 1]
                ensemble_pred = (preds * weights_expanded).sum(dim=0)

                loss = criterion(ensemble_pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                n_batches += 1

            train_loss /= n_batches

            # 驗證
            self.weight_network.eval()
            val_loss = 0.0
            n_batches = 0

            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(self.device), y.to(self.device)

                    x_features = x.mean(dim=-1)
                    weights = self.weight_network(x_features)

                    preds = []
                    for model in self.models:
                        pred = model(x)
                        preds.append(pred)
                    preds = torch.stack(preds, dim=0)

                    weights_expanded = weights.T.unsqueeze(-1).unsqueeze(-1)
                    ensemble_pred = (preds * weights_expanded).sum(dim=0)

                    loss = criterion(ensemble_pred, y)
                    val_loss += loss.item()
                    n_batches += 1

            val_loss /= n_batches

            if val_loss < best_val_loss:
                best_val_loss = val_loss

            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{epochs} - Train: {train_loss:.6f}, Val: {val_loss:.6f}")

        print(f"自適應權重網絡訓練完成 - 最佳驗證 MSE: {best_val_loss:.6f}")

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """自適應集成預測"""
        if self.weight_network is None:
            raise ValueError("需要先訓練權重網絡！")

        x = x.to(self.device)

        # 計算動態權重
        x_features = x.mean(dim=-1)
        weights = self.weight_network(x_features)

        # 獲取預測
        with torch.no_grad():
            preds = []
            for model in self.models:
                pred = model(x)
                preds.append(pred)
            preds = torch.stack(preds, dim=0)

        # 加權組合
        weights_expanded = weights.T.unsqueeze(-1).unsqueeze(-1)
        ensemble_pred = (preds * weights_expanded).sum(dim=0)

        return ensemble_pred
