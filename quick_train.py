#!/usr/bin/env python3
"""
Quick Start Training - One-Click Launch
"""

import sys
import os
from pathlib import Path

# Add SrC to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'SrC'))

from models_PatchTST import PatchTSTConfig
from Training_PatchTST import train_on_data

# Data paths
DATA_DIR = Path("Data/Raw/train_data_neuro")
OUTPUT_DIR = Path("Outputs/quick_start")

print("=" * 60)
print("Neural Tarot Kards v4 - Quick Training")
print("=" * 60)
print()

# Use Affi dataset for quick training
data_paths = [str(DATA_DIR / 'train_data_affi.npz')]

# Quick config (reduced epochs for testing)
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
    max_epochs=10,  # Only 10 epochs for quick test
    early_stop_patience=5
)

save_dir = OUTPUT_DIR
save_dir.mkdir(parents=True, exist_ok=True)

print(f"Training data: train_data_affi.npz")
print(f"Output dir: {save_dir}")
print(f"Epochs: {config.max_epochs}")
print(f"Device: {config.device}")
print()
print("Starting training...")
print("=" * 60)
print()

# Train
model, history = train_on_data(
    data_paths=data_paths,
    config=config,
    save_dir=str(save_dir),
    val_split=0.2
)

print()
print("=" * 60)
print("Training Complete!")
print("=" * 60)
print(f"Model saved to: {save_dir}")
print(f"Final validation MSE: {history['val_mse'][-1]:.6f}")
print()
print("Next steps:")
print(f"1. View results: {save_dir}/training_curves.png")
print("2. Train with more data: python train_competition.py --mode all")
print(f"3. Make predictions: python predict.py --test-data TEST_FILE --models {save_dir}/best_model.pt")
