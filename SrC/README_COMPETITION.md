# Neural Tarot Kards v4 - 比賽解決方案

完整的神經信號預測競賽系統（NSF HDR A3D3 Neural Forecasting Competition）

## 🎯 項目特色

### 核心模型架構
1. **PatchTST** - 基於 Patch 的 Transformer（主力模型）
   - 聚合稀疏神經脈衝，提升語義密度
   - Channel Independence 規避神經元間不穩定相關性
   - RevIN 自動適應分佈偏移（OOD）

2. **iTransformer** - 反轉 Transformer
   - 將每個神經元視為獨立 token
   - 適合多變量獨立預測場景

3. **TimesNet** - 多周期時序建模
   - 2D 變化建模捕捉周期性模式
   - 自適應多尺度聚合

4. **DLinear** - 分解線性模型
   - 趨勢+季節性分解
   - 簡單但極其有效的基線

### 先進技術

#### 數據增強
- 時間抖動（Jitter）
- 時間扭曲（Time Warp）
- 幅度扭曲（Magnitude Warp）
- 頻域遮罩（Frequency Masking）
- MixUp 和 CutMix

#### 特徵工程
- 統計特徵（均值、方差、偏度、峰度等）
- 頻域特徵（FFT、功率譜、頻帶能量）
- 小波特徵（多尺度分解）
- 複雜度特徵（近似熵、Hurst 指數）

#### 模型集成
- 簡單平均
- 加權平均（基於驗證性能）
- Stacking 元學習
- 自適應權重網絡

#### 超參數優化
- 網格搜索
- 隨機搜索
- 貝葉斯優化
- 時間序列交叉驗證

## 📁 項目結構

```
Neural_Tarot_Kards_v4/
├── SrC/                              # 源代碼
│   ├── models_PatchTST/              # PatchTST 模型
│   │   ├── config_PatchTST.py        # 配置
│   │   ├── layers_PatchTST.py        # 層定義
│   │   └── model_PatchTST.py         # 模型定義
│   ├── models_advanced.py            # 先進模型（iTransformer, TimesNet, DLinear）
│   ├── Training_PatchTST.py          # 訓練邏輯
│   ├── data_augmentation.py          # 數據增強
│   ├── ensemble.py                   # 模型集成
│   ├── hyperparameter_tuning.py      # 超參數優化
│   ├── feature_engineering.py        # 特徵工程
│   └── experiment_tracker.py         # 實驗追蹤
├── Data/                             # 數據目錄
│   └── Raw/train_data_neuro/         # 訓練數據
├── Outputs/                          # 輸出目錄
│   ├── competition/                  # 比賽模型
│   └── pipeline/                     # 管道輸出
├── main_PatchTST.py                  # 原始訓練腳本
├── train_competition.py              # 比賽專用訓練腳本
├── train_pipeline.py                 # 統一訓練管道
├── predict.py                        # 預測腳本
└── README_COMPETITION.md             # 本文檔
```

## 🚀 快速開始

### 1. 環境設置

```bash
# 安裝依賴
pip install torch numpy scipy matplotlib tqdm pywt

# （可選）如果使用 CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### 2. 數據準備

確保數據在正確位置：
```
Data/Raw/train_data_neuro/
├── train_data_affi.npz
├── train_data_affi_2024-03-20_private.npz
├── train_data_beignet.npz
├── train_data_beignet_2022-06-01_private.npz
└── train_data_beignet_2022-06-02_private.npz
```

### 3. 訓練模型

#### 選項 A: 快速測試（推薦先運行）

```bash
# 快速訓練測試（20 epochs）
python train_competition.py --mode quick
```

#### 選項 B: 訓練所有配置（完整訓練）

```bash
# 訓練所有數據集和配置（需要較長時間）
python train_competition.py --mode all
```

這將訓練：
- 3 種 PatchTST 配置（v1, v2, deep）
- 7 種數據集組合
- 1 個多尺度集成模型

總共約 22 個模型

#### 選項 C: 只訓練集成模型

```bash
# 僅訓練集成模型
python train_competition.py --mode ensemble
```

#### 選項 D: 使用訓練管道（多模型類型）

```bash
# 訓練多種模型架構並創建集成
python train_pipeline.py \
  --data Data/Raw/train_data_neuro/train_data_affi.npz \
  --models patchtst itransformer timesnet dlinear \
  --augmentation \
  --ensemble \
  --output-dir Outputs/multi_model
```

### 4. 生成預測

```bash
# 使用單個模型預測
python predict.py \
  --test-data path/to/test_data.npz \
  --models Outputs/competition/affi/patchtst_v1/best_model.pt \
  --output predictions.npz

# 使用多個模型集成預測
python predict.py \
  --test-data path/to/test_data.npz \
  --models \
    Outputs/competition/affi/patchtst_v1/best_model.pt \
    Outputs/competition/beignet/patchtst_v1/best_model.pt \
    Outputs/competition/ensemble/multiscale/best_model.pt \
  --ensemble weighted \
  --output ensemble_predictions.npz
```

## 📊 模型配置詳解

### PatchTST v1 (最佳配置)
```python
- patch_len: 16        # 較大的 patch 捕捉長期模式
- stride: 8            # 較大的步長減少計算
- d_model: 256         # 更大的模型容量
- n_heads: 8           # 更多注意力頭
- e_layers: 4          # 中等深度
- dropout: 0.1         # 較低 dropout
- learning_rate: 5e-4  # 較小學習率穩定訓練
```

### PatchTST v2 (平衡配置)
```python
- patch_len: 8         # 標準 patch 大小
- stride: 4            # 標準步長
- d_model: 128         # 中等容量
- n_heads: 4           # 標準注意力頭
- e_layers: 3          # 標準深度
```

### PatchTST Deep (深度配置)
```python
- patch_len: 12        # 中等 patch
- d_model: 192         # 大容量
- e_layers: 5          # 更深的網絡
- dropout: 0.15        # 防止過擬合
```

## 🎓 最佳實踐

### 1. 數據集選擇策略

- **Affi 數據集**: 較新的數據，可能更適合測試集分佈
- **Beignet 數據集**: 較舊但穩定的數據
- **組合策略**:
  - `all_affi`: 所有 Affi 數據（推薦用於 Affi 測試集）
  - `all_beignet`: 所有 Beignet 數據（推薦用於 Beignet 測試集）
  - 混合訓練可能獲得更好的泛化性能

### 2. 集成策略

最佳集成組合（按優先級）：
1. 不同配置的 PatchTST（v1, v2, deep）
2. 不同數據集訓練的同配置模型
3. 不同架構的模型（PatchTST + iTransformer + TimesNet）

推薦集成方法：
- 加權平均（基於驗證 MSE）
- Stacking（如果有足夠的驗證數據）

### 3. 訓練技巧

- **Early Stopping**: 使用驗證集防止過擬合
- **Learning Rate**: 從 1e-3 開始，使用 cosine annealing
- **Data Augmentation**: 適度使用（概率 0.5），不要過度
- **Gradient Clipping**: 設置為 1.0 防止梯度爆炸

### 4. 預測技巧

- **Test-Time Augmentation**: 對測試數據進行多次輕微變換，預測結果平均
- **多模型投票**: 使用 5-10 個不同配置的模型
- **後處理**: 可以使用移動平均平滑預測結果

## 🔬 實驗追蹤

所有訓練實驗會自動記錄在：
```
Outputs/competition/experiments/<experiment_name>/
├── config.json           # 配置
├── history.json          # 訓練歷史
├── training_curves.png   # 訓練曲線
├── predictions.png       # 預測可視化
└── summary.txt          # 總結報告
```

## 📈 性能優化建議

### 如果驗證 MSE 太高：

1. **增加模型容量**:
   - 增大 d_model (128 → 256)
   - 增加層數 e_layers (3 → 5)

2. **調整數據預處理**:
   - 嘗試不同的 gaussian_sigma (0.03 ~ 0.07)
   - 調整 patch_len 和 stride

3. **使用集成**:
   - 至少使用 3 個不同配置的模型
   - 使用加權平均

### 如果過擬合（訓練損失遠小於驗證損失）：

1. **增加正則化**:
   - 增大 dropout (0.1 → 0.3)
   - 增大 weight_decay (1e-5 → 1e-4)

2. **使用數據增強**:
   - 啟用 --augmentation 標誌
   - 調整增強概率

3. **早停**:
   - 減小 early_stop_patience (15 → 10)

## 🏆 比賽提交檢查清單

- [ ] 訓練至少 3 個不同配置的模型
- [ ] 在 Affi 和 Beignet 數據集上都進行訓練
- [ ] 創建模型集成（至少 3 個模型）
- [ ] 驗證預測輸出格式正確
- [ ] 檢查預測值範圍合理（非負數）
- [ ] 保存所有模型和配置
- [ ] 記錄訓練日誌和性能指標

## 🐛 常見問題

### Q: 訓練時內存不足
A: 減小 batch_size (64 → 32 → 16) 或減小模型大小 (d_model)

### Q: 訓練速度太慢
A:
- 使用 GPU (CUDA)
- 減小 max_epochs
- 使用更大的 batch_size（如果內存允許）
- 減小驗證集大小

### Q: 模型不收斂
A:
- 檢查學習率（可能太大）
- 檢查數據是否正確加載
- 嘗試更簡單的模型配置
- 檢查梯度爆炸（啟用 gradient clipping）

## 📧 技術支持

如有問題，請檢查：
1. 訓練日誌 (`experiment.log`)
2. 配置文件 (`config.json`)
3. 訓練曲線 (`training_curves.png`)

---

**祝比賽順利，沖擊第一名！** 🚀
