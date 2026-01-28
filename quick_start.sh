#!/bin/bash
# Neural Tarot Kards v4 - 快速啟動腳本 (Linux/Mac)

echo "========================================"
echo "Neural Tarot Kards v4"
echo "NSF HDR Neural Forecasting Competition"
echo "========================================"
echo ""

echo "檢查 Python 環境..."
python3 --version || python --version
if [ $? -ne 0 ]; then
    echo "錯誤: 未找到 Python！請先安裝 Python 3.8+"
    exit 1
fi
echo ""

echo "檢查依賴..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>/dev/null || python -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo ""
    echo "未安裝依賴！正在安裝..."
    pip install -r requirements.txt || pip3 install -r requirements.txt
fi
echo ""

echo "========================================"
echo "選擇操作:"
echo "========================================"
echo "1. 快速測試訓練 (20 epochs)"
echo "2. 完整訓練 (所有配置)"
echo "3. 僅訓練集成模型"
echo "4. 生成預測"
echo "5. 查看項目結構"
echo "6. 退出"
echo "========================================"
echo ""

read -p "請選擇 (1-6): " choice

case $choice in
    1)
        echo ""
        echo "開始快速測試訓練..."
        python3 train_competition.py --mode quick || python train_competition.py --mode quick
        ;;
    2)
        echo ""
        echo "開始完整訓練（這將需要很長時間）..."
        read -p "確定要繼續嗎？(y/n) " confirm
        if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
            python3 train_competition.py --mode all || python train_competition.py --mode all
        fi
        ;;
    3)
        echo ""
        echo "開始訓練集成模型..."
        python3 train_competition.py --mode ensemble || python train_competition.py --mode ensemble
        ;;
    4)
        echo ""
        echo "預測功能"
        echo "用法: python predict.py --test-data TEST_FILE --models MODEL1 MODEL2 ..."
        echo ""
        echo "示例:"
        echo "python predict.py --test-data Data/test.npz --models Outputs/competition/affi/patchtst_v1/best_model.pt"
        ;;
    5)
        echo ""
        echo "項目結構:"
        tree -L 3 || find . -maxdepth 3 -type d
        ;;
    6)
        echo ""
        echo "再見！"
        exit 0
        ;;
    *)
        echo ""
        echo "無效選擇！"
        ;;
esac

echo ""
echo "========================================"
echo "操作完成！"
echo "========================================"
