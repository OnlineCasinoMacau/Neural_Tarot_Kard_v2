#!/usr/bin/env python3
"""
Neural Tarot Kards v4 - Training Entry Point
=============================================

選擇訓練數據並啟動 PatchTST 訓練
"""

import sys
import os

# 添加 SrC 到路徑
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'SrC'))

from SrC.models_PatchTST import PatchTSTConfig
from SrC.Training_PatchTST import train_on_data

# 數據路徑配置
DATA_DIR = r"C:\Users\goodj\Desktop\NTK\Neural_Tarot_Kards_v4\Data\Raw\train_data_neuro"
OUTPUT_DIR = r"C:\Users\goodj\Desktop\NTK\Neural_Tarot_Kards_v4\Outputs\checkpoints"

# 訓練選項配置
TRAINING_OPTIONS = {
    1: {
        "name": "Train on Affi",
        "files": ["train_data_affi.npz"],
        "output_subdir": "affi"
    },
    2: {
        "name": "Train on Beignet",
        "files": ["train_data_beignet.npz"],
        "output_subdir": "beignet"
    },
    3: {
        "name": "Train on Affi D2",
        "files": ["train_data_affi.npz", "train_data_affi_2024-03-20_private.npz"],
        "output_subdir": "affi_d2"
    },
    4: {
        "name": "Train on Beignet D2 (06-01)",
        "files": ["train_data_beignet.npz", "train_data_beignet_2022-06-01_private.npz"],
        "output_subdir": "beignet_d2_0601"
    },
    5: {
        "name": "Train on Beignet D2 (06-02)",
        "files": ["train_data_beignet.npz", "train_data_beignet_2022-06-02_private.npz"],
        "output_subdir": "beignet_d2_0602"
    }
}


def print_menu():
    """顯示訓練選項菜單"""
    print("\n" + "=" * 60)
    print("  Neural Tarot Kards v4 - PatchTST Training")
    print("=" * 60)
    print("\n請選擇訓練數據:\n")

    for key, option in TRAINING_OPTIONS.items():
        files_str = " + ".join(option["files"])
        print(f"  [{key}] {option['name']}")
        print(f"      -> {files_str}")
        print()

    print("  [0] 退出")
    print("\n" + "-" * 60)


def get_user_choice() -> int:
    """獲取用戶選擇"""
    while True:
        try:
            choice = input("請輸入選項 (0-5): ").strip()
            choice_int = int(choice)
            if choice_int in [0, 1, 2, 3, 4, 5]:
                return choice_int
            else:
                print("無效選項，請輸入 0-5")
        except ValueError:
            print("請輸入數字")


def run_training(option_key: int):
    """執行訓練"""
    option = TRAINING_OPTIONS[option_key]

    print(f"\n選擇: {option['name']}")
    print(f"數據文件: {option['files']}")

    # 構建完整路徑
    data_paths = [os.path.join(DATA_DIR, f) for f in option["files"]]
    save_dir = os.path.join(OUTPUT_DIR, option["output_subdir"])

    # 檢查數據文件是否存在
    for path in data_paths:
        if not os.path.exists(path):
            print(f"\n錯誤: 找不到數據文件 {path}")
            return

    # 創建配置
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
        max_epochs=50,
        early_stop_patience=10,
    )

    # 開始訓練
    model, history = train_on_data(
        data_paths=data_paths,
        config=config,
        save_dir=save_dir,
        val_split=0.2
    )

    return model, history


def main():
    """主程式入口"""
    print_menu()
    choice = get_user_choice()

    if choice == 0:
        print("\n再見!")
        return

    run_training(choice)


if __name__ == "__main__":
    main()
