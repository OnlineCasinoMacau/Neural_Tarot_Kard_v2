#!/usr/bin/env python3
"""
比賽專用訓練腳本
================

使用優化的配置訓練最佳模型
針對 NSF HDR A3D3 Neural Forecasting Competition
"""

import sys
import os
from pathlib import Path
import numpy as np
import torch

# 添加 SrC 到路徑
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'SrC'))

from models_PatchTST import PatchTST, PatchTSTConfig, MultiScaleEnsemble
from Training_PatchTST import train_on_data, load_npz_data

# 數據路徑
DATA_DIR = Path("Data/Raw/train_data_neuro")
OUTPUT_DIR = Path("Outputs/competition")

# 最佳超參數配置（基於文獻和實驗）
BEST_CONFIGS = {
    'patchtst_v1': PatchTSTConfig(
        input_size=96,
        horizon=24,
        patch_len=16,
        stride=8,
        d_model=256,
        n_heads=8,
        e_layers=4,
        d_ff=512,
        dropout=0.1,
        revin=True,
        affine=True,
        batch_size=64,
        learning_rate=5e-4,
        weight_decay=1e-5,
        max_epochs=100,
        early_stop_patience=15,
        gradient_clip=1.0,
        mse_weight=1.0,
        mae_weight=0.3,
        gaussian_sigma=0.05
    ),

    'patchtst_v2': PatchTSTConfig(
        input_size=96,
        horizon=24,
        patch_len=8,
        stride=4,
        d_model=128,
        n_heads=4,
        e_layers=3,
        d_ff=256,
        dropout=0.2,
        revin=True,
        affine=False,
        batch_size=32,
        learning_rate=1e-3,
        weight_decay=5e-5,
        max_epochs=80,
        early_stop_patience=12,
        gradient_clip=1.0,
        mse_weight=1.0,
        mae_weight=0.5,
        gaussian_sigma=0.03
    ),

    'patchtst_deep': PatchTSTConfig(
        input_size=96,
        horizon=24,
        patch_len=12,
        stride=6,
        d_model=192,
        n_heads=6,
        e_layers=5,
        d_ff=384,
        dropout=0.15,
        revin=True,
        affine=True,
        batch_size=48,
        learning_rate=8e-4,
        weight_decay=1e-5,
        max_epochs=120,
        early_stop_patience=18,
        gradient_clip=1.0,
        mse_weight=1.0,
        mae_weight=0.4,
        gaussian_sigma=0.04
    ),
}


def train_all_datasets():
    """在所有數據集上訓練模型"""

    datasets = {
        'affi': ['train_data_affi.npz'],
        'affi_d2': ['train_data_affi.npz', 'train_data_affi_2024-03-20_private.npz'],
        'beignet': ['train_data_beignet.npz'],
        'beignet_0601': ['train_data_beignet.npz', 'train_data_beignet_2022-06-01_private.npz'],
        'beignet_0602': ['train_data_beignet.npz', 'train_data_beignet_2022-06-02_private.npz'],
        'all_affi': [
            'train_data_affi.npz',
            'train_data_affi_2024-03-20_private.npz'
        ],
        'all_beignet': [
            'train_data_beignet.npz',
            'train_data_beignet_2022-06-01_private.npz',
            'train_data_beignet_2022-06-02_private.npz'
        ],
    }

    results = {}

    for dataset_name, file_list in datasets.items():
        print("\n" + "=" * 80)
        print(f"訓練數據集: {dataset_name}")
        print("=" * 80)

        # 構建完整路徑
        data_paths = [str(DATA_DIR / f) for f in file_list]

        # 訓練多個配置
        for config_name, config in BEST_CONFIGS.items():
            save_dir = OUTPUT_DIR / dataset_name / config_name
            save_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n訓練配置: {config_name}")
            print(f"保存目錄: {save_dir}")

            try:
                model, history = train_on_data(
                    data_paths=data_paths,
                    config=config,
                    save_dir=str(save_dir),
                    val_split=0.2
                )

                # 記錄結果
                best_mse = min(history['val_mse']) if 'val_mse' in history else float('inf')
                results[f"{dataset_name}_{config_name}"] = {
                    'best_mse': best_mse,
                    'best_epoch': history.get('best_epoch', 0),
                    'final_mse': history['val_mse'][-1] if 'val_mse' in history else None
                }

                print(f"✓ {config_name} 訓練完成 - 最佳 MSE: {best_mse:.6f}")

            except Exception as e:
                print(f"✗ {config_name} 訓練失敗: {e}")
                results[f"{dataset_name}_{config_name}"] = {'error': str(e)}

    return results


def train_ensemble_models():
    """訓練集成模型"""

    print("\n" + "=" * 80)
    print("訓練多尺度集成模型")
    print("=" * 80)

    # 組合所有數據
    all_data_paths = [
        str(DATA_DIR / 'train_data_affi.npz'),
        str(DATA_DIR / 'train_data_affi_2024-03-20_private.npz'),
        str(DATA_DIR / 'train_data_beignet.npz'),
        str(DATA_DIR / 'train_data_beignet_2022-06-01_private.npz'),
        str(DATA_DIR / 'train_data_beignet_2022-06-02_private.npz'),
    ]

    # 使用最佳配置訓練多尺度集成
    base_config = BEST_CONFIGS['patchtst_v1']

    save_dir = OUTPUT_DIR / 'ensemble' / 'multiscale'
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"保存目錄: {save_dir}")

    try:
        model, history = train_on_data(
            data_paths=all_data_paths,
            config=base_config,
            save_dir=str(save_dir),
            val_split=0.15  # 使用更多數據訓練
        )

        best_mse = min(history['val_mse']) if 'val_mse' in history else float('inf')
        print(f"✓ 集成模型訓練完成 - 最佳 MSE: {best_mse:.6f}")

        return best_mse

    except Exception as e:
        print(f"✗ 集成模型訓練失敗: {e}")
        return None


def print_results_summary(results):
    """打印結果總結"""

    print("\n" + "=" * 80)
    print("訓練結果總結")
    print("=" * 80)

    # 按 MSE 排序
    sorted_results = sorted(
        [(k, v) for k, v in results.items() if 'best_mse' in v],
        key=lambda x: x[1]['best_mse']
    )

    print(f"\n{'模型配置':<40} | {'最佳 MSE':>12} | {'最佳 Epoch':>10}")
    print("-" * 80)

    for name, result in sorted_results[:20]:  # 顯示前 20 個
        mse = result['best_mse']
        epoch = result.get('best_epoch', 'N/A')
        print(f"{name:<40} | {mse:>12.6f} | {epoch:>10}")

    # 保存結果
    import json
    results_file = OUTPUT_DIR / 'training_results.json'
    with open(results_file, 'w') as f:
        # 轉換為可序列化格式
        serializable_results = {}
        for k, v in results.items():
            serializable_results[k] = {
                key: (float(val) if isinstance(val, (np.floating, np.integer)) else val)
                for key, val in v.items()
            }
        json.dump(serializable_results, f, indent=2)

    print(f"\n結果已保存至: {results_file}")


def quick_train():
    """快速訓練腳本 - 用於測試"""

    print("\n" + "=" * 80)
    print("快速訓練模式（測試用）")
    print("=" * 80)

    # 使用 Affi 數據集快速訓練
    data_paths = [str(DATA_DIR / 'train_data_affi.npz')]

    # 快速配置（較少 epoch）
    quick_config = PatchTSTConfig(
        input_size=96,
        horizon=24,
        patch_len=8,
        stride=4,
        d_model=128,
        n_heads=4,
        e_layers=3,
        dropout=0.2,
        revin=True,
        batch_size=32,
        learning_rate=1e-3,
        max_epochs=20,  # 減少 epoch
        early_stop_patience=5
    )

    save_dir = OUTPUT_DIR / 'quick_test'
    save_dir.mkdir(parents=True, exist_ok=True)

    model, history = train_on_data(
        data_paths=data_paths,
        config=quick_config,
        save_dir=str(save_dir),
        val_split=0.2
    )

    print("\n快速訓練完成！")
    print(f"最終 MSE: {history['val_mse'][-1]:.6f}")


def main():
    """主函數"""

    import argparse

    parser = argparse.ArgumentParser(description='Neural Forecasting Competition - Training')

    parser.add_argument(
        '--mode',
        type=str,
        default='all',
        choices=['all', 'ensemble', 'quick'],
        help='訓練模式'
    )

    parser.add_argument(
        '--dataset',
        type=str,
        choices=['affi', 'beignet', 'all'],
        help='指定單個數據集'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Neural Forecasting Competition - 專業訓練腳本")
    print("=" * 80)
    print(f"輸出目錄: {OUTPUT_DIR}")
    print(f"設備: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("=" * 80)

    if args.mode == 'quick':
        quick_train()

    elif args.mode == 'ensemble':
        train_ensemble_models()

    elif args.mode == 'all':
        # 訓練所有配置
        results = train_all_datasets()

        # 訓練集成
        ensemble_mse = train_ensemble_models()
        if ensemble_mse is not None:
            results['ensemble_multiscale'] = {'best_mse': ensemble_mse}

        # 打印總結
        print_results_summary(results)

    print("\n" + "=" * 80)
    print("訓練流程完成！")
    print("=" * 80)
    print(f"\n模型已保存至: {OUTPUT_DIR}")
    print("\n下一步:")
    print("1. 查看訓練結果: cat Outputs/competition/training_results.json")
    print("2. 使用最佳模型進行預測:")
    print("   python predict.py --test-data <test_file> --models Outputs/competition/*/best_model.pt")


if __name__ == '__main__':
    main()
