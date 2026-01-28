#!/usr/bin/env python3
"""
實驗追蹤和可視化
=================

記錄實驗結果並生成可視化報告
"""

import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import matplotlib.pyplot as plt
import numpy as np


class ExperimentTracker:
    """實驗追蹤器"""

    def __init__(self, experiment_name: str, base_dir: str = './experiments'):
        self.experiment_name = experiment_name
        self.base_dir = Path(base_dir)
        self.exp_dir = self.base_dir / experiment_name / datetime.now().strftime('%Y%m%d_%H%M%S')
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        self.config = {}
        self.metrics = {}
        self.logs = []

        print(f"實驗目錄: {self.exp_dir}")

    def log_config(self, config: Dict):
        """記錄實驗配置"""
        self.config = config

        config_path = self.exp_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"配置已保存: {config_path}")

    def log_metric(self, name: str, value: float, step: Optional[int] = None):
        """記錄單個指標"""
        if name not in self.metrics:
            self.metrics[name] = []

        entry = {'value': value}
        if step is not None:
            entry['step'] = step

        self.metrics[name].append(entry)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """批量記錄指標"""
        for name, value in metrics.items():
            self.log_metric(name, value, step)

    def log_history(self, history: Dict[str, List[float]]):
        """記錄訓練歷史"""
        history_path = self.exp_dir / 'history.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

        print(f"訓練歷史已保存: {history_path}")

        # 生成訓練曲線
        self.plot_training_curves(history)

    def log_text(self, message: str, level: str = 'INFO'):
        """記錄文本日誌"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] [{level}] {message}"

        self.logs.append(log_entry)
        print(log_entry)

        # 寫入日誌文件
        log_path = self.exp_dir / 'experiment.log'
        with open(log_path, 'a') as f:
            f.write(log_entry + '\n')

    def save_model_info(self, model_info: Dict):
        """保存模型信息"""
        model_info_path = self.exp_dir / 'model_info.json'
        with open(model_info_path, 'w') as f:
            json.dump(model_info, f, indent=2)

        print(f"模型信息已保存: {model_info_path}")

    def plot_training_curves(self, history: Dict[str, List[float]]):
        """繪製訓練曲線"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{self.experiment_name} - Training Curves', fontsize=16)

        # 訓練損失
        if 'train_loss' in history:
            axes[0, 0].plot(history['train_loss'], label='Train Loss')
            if 'val_loss' in history:
                axes[0, 0].plot(history['val_loss'], label='Val Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Training and Validation Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

        # 驗證 MSE
        if 'val_mse' in history:
            axes[0, 1].plot(history['val_mse'], label='Val MSE', color='orange')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('MSE')
            axes[0, 1].set_title('Validation MSE')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

        # 學習率（如果有）
        if 'learning_rate' in history:
            axes[1, 0].plot(history['learning_rate'], label='Learning Rate', color='green')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_title('Learning Rate Schedule')
            axes[1, 0].set_yscale('log')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

        # 其他指標
        other_metrics = [k for k in history.keys()
                        if k not in ['train_loss', 'val_loss', 'val_mse', 'learning_rate']]
        if other_metrics:
            for metric in other_metrics[:3]:  # 最多顯示 3 個
                axes[1, 1].plot(history[metric], label=metric)
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Value')
            axes[1, 1].set_title('Other Metrics')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        plot_path = self.exp_dir / 'training_curves.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"訓練曲線已保存: {plot_path}")

    def plot_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_samples: int = 5,
        n_channels: int = 3
    ):
        """
        繪製預測結果對比

        Args:
            y_true: 真實值 [n_samples, horizon, n_channels]
            y_pred: 預測值 [n_samples, horizon, n_channels]
            n_samples: 顯示的樣本數
            n_channels: 顯示的通道數
        """
        n_samples = min(n_samples, y_true.shape[0])
        n_channels = min(n_channels, y_true.shape[2])

        fig, axes = plt.subplots(n_samples, n_channels, figsize=(15, 3*n_samples))
        if n_samples == 1:
            axes = axes.reshape(1, -1)

        fig.suptitle(f'{self.experiment_name} - Prediction Results', fontsize=16)

        for i in range(n_samples):
            for j in range(n_channels):
                ax = axes[i, j] if n_samples > 1 else axes[j]

                ax.plot(y_true[i, :, j], label='True', linewidth=2, alpha=0.7)
                ax.plot(y_pred[i, :, j], label='Predicted', linewidth=2, alpha=0.7, linestyle='--')

                mse = np.mean((y_true[i, :, j] - y_pred[i, :, j]) ** 2)
                ax.set_title(f'Sample {i+1}, Channel {j+1} (MSE: {mse:.4f})')
                ax.legend()
                ax.grid(True, alpha=0.3)

        plt.tight_layout()

        plot_path = self.exp_dir / 'predictions.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"預測結果已保存: {plot_path}")

    def generate_summary(self) -> str:
        """生成實驗總結報告"""
        summary = []
        summary.append("=" * 60)
        summary.append(f"實驗名稱: {self.experiment_name}")
        summary.append(f"時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary.append("=" * 60)
        summary.append("\n配置:")
        summary.append(json.dumps(self.config, indent=2))

        summary.append("\n最終指標:")
        for name, values in self.metrics.items():
            if values:
                latest = values[-1]['value']
                summary.append(f"  {name}: {latest:.6f}")

        summary_text = "\n".join(summary)

        # 保存總結
        summary_path = self.exp_dir / 'summary.txt'
        with open(summary_path, 'w') as f:
            f.write(summary_text)

        print(f"\n實驗總結已保存: {summary_path}")
        return summary_text

    def save_results_csv(self, results: List[Dict[str, Any]], filename: str = 'results.csv'):
        """保存結果為 CSV"""
        csv_path = self.exp_dir / filename

        if not results:
            print("沒有結果可保存")
            return

        # 獲取所有鍵
        keys = set()
        for result in results:
            keys.update(result.keys())

        keys = sorted(keys)

        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)

        print(f"結果已保存至 CSV: {csv_path}")


class ComparisonReport:
    """模型比較報告生成器"""

    def __init__(self, experiments: List[ExperimentTracker]):
        self.experiments = experiments

    def generate_comparison_table(self) -> str:
        """生成模型比較表格"""
        table = []
        table.append("模型比較")
        table.append("=" * 80)

        headers = ["實驗名稱", "最終訓練損失", "最終驗證損失", "最佳驗證 MSE"]
        table.append(" | ".join(f"{h:20s}" for h in headers))
        table.append("-" * 80)

        for exp in self.experiments:
            name = exp.experiment_name
            train_loss = exp.metrics.get('train_loss', [{'value': 0}])[-1]['value']
            val_loss = exp.metrics.get('val_loss', [{'value': 0}])[-1]['value']
            val_mse = exp.metrics.get('val_mse', [{'value': 0}])[-1]['value']

            row = [
                name[:20],
                f"{train_loss:.6f}",
                f"{val_loss:.6f}",
                f"{val_mse:.6f}"
            ]
            table.append(" | ".join(f"{r:20s}" for r in row))

        return "\n".join(table)

    def plot_comparison(self, metric: str = 'val_mse'):
        """繪製多個實驗的對比圖"""
        fig, ax = plt.subplots(figsize=(12, 6))

        for exp in self.experiments:
            if metric in exp.metrics:
                values = [m['value'] for m in exp.metrics[metric]]
                ax.plot(values, label=exp.experiment_name, linewidth=2, alpha=0.7)

        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric)
        ax.set_title(f'Model Comparison - {metric}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'comparison_{metric}.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"比較圖已保存: comparison_{metric}.png")


class ProgressMonitor:
    """訓練進度監控器"""

    def __init__(self, total_epochs: int):
        self.total_epochs = total_epochs
        self.current_epoch = 0
        self.metrics_history = []

    def update(self, epoch: int, metrics: Dict[str, float]):
        """更新進度"""
        self.current_epoch = epoch
        self.metrics_history.append(metrics)

        # 估計剩餘時間（簡化版）
        progress = (epoch + 1) / self.total_epochs * 100

        print(f"\nEpoch [{epoch+1}/{self.total_epochs}] - 進度: {progress:.1f}%")
        for name, value in metrics.items():
            print(f"  {name}: {value:.6f}")

        # 檢查是否改善
        if len(self.metrics_history) > 1 and 'val_loss' in metrics:
            prev_val_loss = self.metrics_history[-2].get('val_loss', float('inf'))
            curr_val_loss = metrics['val_loss']

            if curr_val_loss < prev_val_loss:
                improvement = (prev_val_loss - curr_val_loss) / prev_val_loss * 100
                print(f"  ✓ 驗證損失改善: {improvement:.2f}%")
            else:
                print("  ✗ 驗證損失未改善")

    def get_summary(self) -> str:
        """獲取訓練總結"""
        if not self.metrics_history:
            return "無訓練記錄"

        summary = []
        summary.append("\n訓練總結:")
        summary.append("-" * 40)

        # 最佳指標
        if 'val_loss' in self.metrics_history[0]:
            val_losses = [m['val_loss'] for m in self.metrics_history]
            best_epoch = np.argmin(val_losses)
            best_loss = val_losses[best_epoch]

            summary.append(f"最佳 Epoch: {best_epoch + 1}")
            summary.append(f"最佳驗證損失: {best_loss:.6f}")

        return "\n".join(summary)
