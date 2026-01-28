# Neural Tarot Kards v4

Complete solution for NSF HDR A3D3 Neural Forecasting Competition.

## 🚀 Quick Start

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Train Models

**Option 1: Interactive Training (Recommended for beginners)**
```bash
python main_PatchTST.py
```
Select dataset from menu (1-5).

**Option 2: Full Competition Training (Best performance)**
```bash
python train_competition.py --mode all
```
Trains 22 models across all datasets and configurations.
**Time: 3-5 days on RTX 4060**

**Option 3: Training Pipeline (Multi-model architectures)**
```bash
python train_pipeline.py \
  --data Data/Raw/train_data_neuro/train_data_affi.npz \
  --models patchtst itransformer timesnet dlinear \
  --ensemble \
  --output-dir Outputs/pipeline
```

### Generate Predictions
```bash
python predict.py \
  --test-data path/to/test.npz \
  --models model1.pt model2.pt model3.pt \
  --ensemble weighted \
  --output predictions.npz
```

## 📁 Project Structure

```
Neural_Tarot_Kards_v4/
├── SrC/                              # Core modules
│   ├── models_PatchTST/              # PatchTST implementation
│   ├── models_advanced.py            # iTransformer, TimesNet, DLinear
│   ├── Training_PatchTST.py          # Training logic
│   ├── data_augmentation.py          # 10+ augmentation techniques
│   ├── ensemble.py                   # Model ensemble framework
│   ├── hyperparameter_tuning.py      # Grid/Random/Bayesian search
│   ├── feature_engineering.py        # Feature extraction
│   └── experiment_tracker.py         # Logging and visualization
├── main_PatchTST.py                  # Interactive training menu
├── train_competition.py              # Full competition training
├── train_pipeline.py                 # Unified training pipeline
├── predict.py                        # Inference interface
├── README.md                         # This file
├── README_COMPETITION.md             # Detailed documentation
├── PROJECT_SUMMARY.md                # Technical summary
└── requirements.txt                  # Dependencies
```

## 🎯 Model Architectures

1. **PatchTST** - Patch-based Transformer (main model)
2. **iTransformer** - Inverted Transformer
3. **TimesNet** - Multi-period modeling
4. **DLinear** - Decomposition linear baseline

## 📊 Training Time Estimates (RTX 4060)

| Training Mode | Time | Models |
|--------------|------|--------|
| Interactive (single dataset) | 7-11 hours | 1 |
| Full competition training | 3-5 days | 22 |
| Pipeline (4 architectures) | 12-24 hours | 4 |

## 📖 Documentation

- **[README_COMPETITION.md](README_COMPETITION.md)** - Complete usage guide
- **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** - Technical details and features

## 🏆 Competition Strategy

For best results:
1. Train all configurations: `python train_competition.py --mode all`
2. Use ensemble of top 3-5 models
3. Apply weighted averaging based on validation MSE

## 📝 License

Developed for NSF HDR A3D3 Neural Forecasting Competition.

---

**Good luck with the competition!** 🚀
