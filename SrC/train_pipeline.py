#!/usr/bin/env python3
"""
統一訓練管道
=============

自動化訓練流程，支援：
- 多個模型架構
- 交叉驗證
- 超參數搜索
- 模型集成
- 實驗追蹤
"""

import sys
import os
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 添加 SrC 到路徑
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'SrC'))

from models_PatchTST import PatchTST, PatchTSTConfig
from models_advanced import iTransformer, TimesNet, DLinear, ModelFactory
from Training_PatchTST import (
    NeuralDataset,
    PatchTSTTrainer,
    load_npz_data,
    train_on_data
)
from ensemble import EnsemblePredictor
from experiment_tracker import ExperimentTracker
from data_augmentation import TimeSeriesAugmenter, AugmentedDataset


class TrainingPipeline:
    """訓練管道"""

    def __init__(
        self,
        data_paths: list,
        output_dir: str = './outputs/pipeline',
        experiment_name: str = 'neural_forecasting'
    ):
        self.data_paths = data_paths
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.experiment_name = experiment_name
        self.tracker = ExperimentTracker(
            experiment_name=experiment_name,
            base_dir=str(self.output_dir / 'experiments')
        )

        self.trained_models = []
        self.model_performances = []

    def load_data(self, val_split: float = 0.2):
        """加載和劃分數據"""
        print("\n" + "=" * 60)
        print("載入數據...")
        print("=" * 60)

        # 加載數據
        self.data = load_npz_data(self.data_paths)
        print(f"總數據形狀: {self.data.shape}")

        # 劃分訓練/驗證集
        split_idx = int(len(self.data) * (1 - val_split))
        self.train_data = self.data[:split_idx]
        self.val_data = self.data[split_idx:]

        print(f"訓練集: {self.train_data.shape}")
        print(f"驗證集: {self.val_data.shape}")

        self.tracker.log_text(f"數據加載完成 - 訓練: {self.train_data.shape}, 驗證: {self.val_data.shape}")

    def create_dataloaders(
        self,
        config: dict,
        use_augmentation: bool = False,
        augment_prob: float = 0.5
    ):
        """創建數據加載器"""
        train_dataset = NeuralDataset(
            self.train_data,
            input_size=config['input_size'],
            horizon=config['horizon'],
            transform=True
        )

        val_dataset = NeuralDataset(
            self.val_data,
            input_size=config['input_size'],
            horizon=config['horizon'],
            transform=True
        )

        # 數據增強
        if use_augmentation:
            augmenter = TimeSeriesAugmenter()
            train_dataset = AugmentedDataset(
                train_dataset,
                augmenter=augmenter,
                augment_prob=augment_prob,
                n_augment_ops=2
            )
            print(f"✓ 啟用數據增強 (概率: {augment_prob})")

        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=0
        )

        return train_loader, val_loader

    def train_single_model(
        self,
        model_type: str,
        config: dict,
        model_name: str = None,
        use_augmentation: bool = False
    ):
        """訓練單個模型"""
        if model_name is None:
            model_name = f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        print("\n" + "=" * 60)
        print(f"訓練模型: {model_name}")
        print("=" * 60)

        self.tracker.log_text(f"開始訓練 {model_name}")
        self.tracker.log_config(config)

        # 創建數據加載器
        train_loader, val_loader = self.create_dataloaders(
            config,
            use_augmentation=use_augmentation
        )

        # 創建模型
        if model_type.lower() == 'patchtst':
            model_config = PatchTSTConfig(**config)
            model = PatchTST(model_config)
        elif model_type.lower() == 'itransformer':
            model = iTransformer(**config)
        elif model_type.lower() == 'timesnet':
            model = TimesNet(**config)
        elif model_type.lower() == 'dlinear':
            model = DLinear(**config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # 創建訓練器
        if model_type.lower() == 'patchtst':
            trainer = PatchTSTTrainer(
                model=model,
                config=model_config,
                save_dir=str(self.output_dir / model_name)
            )
        else:
            # 為其他模型創建通用訓練器
            trainer = self._create_generic_trainer(model, config, model_name)

        # 訓練
        history = trainer.fit(train_loader, val_loader)

        # 記錄歷史
        self.tracker.log_history(history)

        # 評估
        final_val_loss, final_val_mse = trainer.validate(val_loader)
        self.tracker.log_metric('final_val_loss', final_val_loss)
        self.tracker.log_metric('final_val_mse', final_val_mse)

        print(f"\n{model_name} 訓練完成!")
        print(f"最終驗證 MSE: {final_val_mse:.6f}")

        # 保存模型信息
        self.trained_models.append({
            'name': model_name,
            'type': model_type,
            'model': model,
            'trainer': trainer,
            'config': config,
            'path': str(self.output_dir / model_name / 'best_model.pt')
        })

        self.model_performances.append({
            'name': model_name,
            'type': model_type,
            'val_mse': final_val_mse,
            'val_loss': final_val_loss
        })

        return model, history

    def _create_generic_trainer(self, model, config, model_name):
        """為非 PatchTST 模型創建通用訓練器"""
        # 簡化版訓練器
        class GenericTrainer:
            def __init__(self, model, config, save_dir):
                self.model = model.to(config.get('device', 'cpu'))
                self.config = config
                self.save_dir = Path(save_dir)
                self.save_dir.mkdir(parents=True, exist_ok=True)

                self.optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=config.get('learning_rate', 1e-3),
                    weight_decay=config.get('weight_decay', 1e-4)
                )

                self.criterion = nn.MSELoss()
                self.history = {
                    'train_loss': [],
                    'val_loss': [],
                    'val_mse': [],
                    'best_val_loss': float('inf'),
                    'best_epoch': 0
                }

            def fit(self, train_loader, val_loader):
                from tqdm import tqdm

                for epoch in range(self.config.get('max_epochs', 50)):
                    # 訓練
                    self.model.train()
                    train_loss = 0.0

                    for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
                        x = x.to(self.config.get('device', 'cpu'))
                        y = y.to(self.config.get('device', 'cpu'))

                        pred = self.model(x)
                        loss = self.criterion(pred, y)

                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

                        train_loss += loss.item()

                    train_loss /= len(train_loader)
                    self.history['train_loss'].append(train_loss)

                    # 驗證
                    val_loss, val_mse = self.validate(val_loader)
                    self.history['val_loss'].append(val_loss)
                    self.history['val_mse'].append(val_mse)

                    print(f"Epoch {epoch+1} - Train: {train_loss:.4f}, Val: {val_loss:.4f}, MSE: {val_mse:.4f}")

                    # 保存最佳模型
                    if val_loss < self.history['best_val_loss']:
                        self.history['best_val_loss'] = val_loss
                        self.history['best_epoch'] = epoch
                        self.save_model('best_model.pt')

                return self.history

            def validate(self, val_loader):
                self.model.eval()
                total_loss = 0.0
                total_mse = 0.0

                with torch.no_grad():
                    for x, y in val_loader:
                        x = x.to(self.config.get('device', 'cpu'))
                        y = y.to(self.config.get('device', 'cpu'))

                        pred = self.model(x)
                        loss = self.criterion(pred, y)
                        mse = nn.MSELoss()(pred, y)

                        total_loss += loss.item()
                        total_mse += mse.item()

                return total_loss / len(val_loader), total_mse / len(val_loader)

            def save_model(self, filename):
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'history': self.history
                }, self.save_dir / filename)

            def load_model(self, filename):
                checkpoint = torch.load(self.save_dir / filename)
                self.model.load_state_dict(checkpoint['model_state_dict'])

        return GenericTrainer(model, config, str(self.output_dir / model_name))

    def train_multiple_models(
        self,
        model_configs: list,
        use_augmentation: bool = False
    ):
        """訓練多個模型"""
        print("\n" + "=" * 60)
        print(f"訓練 {len(model_configs)} 個模型...")
        print("=" * 60)

        for i, (model_type, config) in enumerate(model_configs):
            model_name = f"{model_type}_{i+1}"
            self.train_single_model(
                model_type=model_type,
                config=config,
                model_name=model_name,
                use_augmentation=use_augmentation
            )

        # 打印性能總結
        print("\n" + "=" * 60)
        print("所有模型性能總結:")
        print("=" * 60)
        for perf in sorted(self.model_performances, key=lambda x: x['val_mse']):
            print(f"{perf['name']:30s} | MSE: {perf['val_mse']:.6f} | Loss: {perf['val_loss']:.6f}")

    def create_ensemble(self, method: str = 'weighted', save_dir: str = None):
        """創建集成模型"""
        if save_dir is None:
            save_dir = str(self.output_dir / 'ensemble')

        print("\n" + "=" * 60)
        print(f"創建集成模型 (方法: {method})...")
        print("=" * 60)

        # 提取模型
        models = [m['model'] for m in self.trained_models]

        # 創建集成
        ensemble = EnsemblePredictor(
            models=models,
            device=self.trained_models[0]['config'].get('device', 'cpu')
        )

        # 創建驗證數據加載器
        _, val_loader = self.create_dataloaders(self.trained_models[0]['config'])

        # 計算最優權重
        if method == 'weighted':
            weights = ensemble.compute_optimal_weights(val_loader, method='inverse_loss')

        # 保存集成
        ensemble.save(save_dir)

        print(f"集成模型已保存至: {save_dir}")
        return ensemble


def main():
    parser = argparse.ArgumentParser(description='Neural Forecasting - Training Pipeline')

    parser.add_argument(
        '--data',
        type=str,
        nargs='+',
        required=True,
        help='訓練數據文件路徑'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='./Outputs/pipeline',
        help='輸出目錄'
    )

    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        default=['patchtst'],
        choices=['patchtst', 'itransformer', 'timesnet', 'dlinear'],
        help='要訓練的模型類型'
    )

    parser.add_argument(
        '--augmentation',
        action='store_true',
        help='啟用數據增強'
    )

    parser.add_argument(
        '--ensemble',
        action='store_true',
        help='創建集成模型'
    )

    parser.add_argument(
        '--experiment-name',
        type=str,
        default='neural_forecasting',
        help='實驗名稱'
    )

    args = parser.parse_args()

    # 創建管道
    pipeline = TrainingPipeline(
        data_paths=args.data,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name
    )

    # 加載數據
    pipeline.load_data(val_split=0.2)

    # 準備模型配置
    base_config = {
        'input_size': 96,
        'horizon': 24,
        'd_model': 128,
        'n_heads': 4,
        'e_layers': 3,
        'dropout': 0.2,
        'batch_size': 32,
        'learning_rate': 1e-3,
        'max_epochs': 50,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    # 為每個模型類型創建配置
    model_configs = []
    for model_type in args.models:
        if model_type == 'patchtst':
            config = {**base_config, 'patch_len': 8, 'stride': 4, 'revin': True}
        else:
            config = base_config.copy()

        model_configs.append((model_type, config))

    # 訓練模型
    pipeline.train_multiple_models(
        model_configs=model_configs,
        use_augmentation=args.augmentation
    )

    # 創建集成
    if args.ensemble and len(pipeline.trained_models) > 1:
        pipeline.create_ensemble(method='weighted')

    # 生成總結
    pipeline.tracker.generate_summary()

    print("\n訓練管道完成！")


if __name__ == '__main__':
    main()
