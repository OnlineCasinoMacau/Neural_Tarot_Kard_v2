# Neural Tarot Kards v4

NSF HDR A3D3 Neural Forecasting Competition Solution.

## 🚀 Quick Start

```bash
python main_PatchTST.py
```

Select dataset from interactive menu (options 1-5).

## 📁 Project Structure

```
Neural_Tarot_Kards_v4/
├── main_PatchTST.py              # Main training script (start here)
├── Data/                         # Training data
│   └── Raw/train_data_neuro/
├── Outputs/                      # Model outputs
│   └── checkpoints/
└── SrC/                          # All source code & docs
    ├── README.md                 # Full documentation
    ├── README_COMPETITION.md     # Competition guide
    ├── PROJECT_SUMMARY.md        # Technical summary
    ├── requirements.txt          # Dependencies
    ├── models_PatchTST/          # PatchTST model
    ├── models_advanced.py        # Advanced models
    ├── Training_PatchTST.py      # Training logic
    ├── train_competition.py      # Full competition training
    ├── train_pipeline.py         # Training pipeline
    ├── predict.py                # Inference
    ├── data_augmentation.py      # Data augmentation
    ├── ensemble.py               # Model ensemble
    ├── hyperparameter_tuning.py  # HPO
    ├── feature_engineering.py    # Feature extraction
    ├── experiment_tracker.py     # Logging
    └── Configs/                  # Configuration files
```

## 📖 Documentation

See **[SrC/README.md](SrC/README.md)** for complete documentation.

## ⏱️ Training Time (RTX 4060)

- Single dataset: 7-11 hours
- All datasets: 3-5 days

## 🏆 Competition

For full competition training:
```bash
cd SrC
python train_competition.py --mode all
```

---

**Repository:** https://github.com/OnlineCasinoMacau/Neural_Tarot_Kard_v2
