#!/usr/bin/env python3
"""
快速開始訓練 - 一鍵啟動
"""

import sys
import os
from pathlib import Path

# 添加 SrC 到路徑
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'SrC'))

from models_PatchTST import PatchTSTConfig
from Training_PatchTST import train_on_data

# 數據路徑
DATA_DIR = Path("Data/Raw/train_data_neuro")
OUTPUT_DIR = Path("Outputs/quick_start")

print("=" * 60)
print("Neural Tarot Kards v4 - 快速訓練")
print("=" * 60)
print()

# 使用 Affi 數據集快速訓練
data_paths = [str(DATA_DIR / 'train_data_affi.npz')]

# 快速配置（減少 epochs 用於測試）
config = PatchTSTConfig(
    input_size=96,
    horizon=24,
    patch_len=8,
    stride=4,
    d_model=128,
    n_heads=4,
    e_layers=3,
    dropout=0.2,
    revin=True,
    affine=False,
    batch_size=32,
    learning_rate=1e-3,
    max_epochs=10,  # 只訓練 10 個 epoch 用於快速測試
    early_stop_patience=5
)

save_dir = OUTPUT_DIR
save_dir.mkdir(parents=True, exist_ok=True)

print(f"訓練數據: train_data_affi.npz")
print(f"輸出目錄: {save_dir}")
print(f"訓練輪數: {config.max_epochs}")
print(f"設備: {config.device}")
print()
print("開始訓練...")
print("=" * 60)
print()

# 訓練
model, history = train_on_data(
    data_paths=data_paths,
    config=config,
    save_dir=str(save_dir),
    val_split=0.2
)

print()
print("=" * 60)
print("訓練完成！")
print("=" * 60)
print(f"模型保存位置: {save_dir}")
print(f"最終驗證 MSE: {history['val_mse'][-1]:.6f}")
print()
print("下一步:")
print("1. 查看訓練結果: {}/training_curves.png".format(save_dir))
print("2. 使用更多數據訓練: python train_competition.py --mode all")
print("3. 生成預測: python predict.py --test-data TEST_FILE --models {}/best_model.pt".format(save_dir))
