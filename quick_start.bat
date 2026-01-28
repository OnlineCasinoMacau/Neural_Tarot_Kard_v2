@echo off
REM Neural Tarot Kards v4 - 快速啟動腳本 (Windows)

echo ========================================
echo Neural Tarot Kards v4
echo NSF HDR Neural Forecasting Competition
echo ========================================
echo.

echo 檢查 Python 環境...
python --version
if %errorlevel% neq 0 (
    echo 錯誤: 未找到 Python！請先安裝 Python 3.8+
    pause
    exit /b 1
)
echo.

echo 檢查依賴...
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
if %errorlevel% neq 0 (
    echo.
    echo 未安裝依賴！正在安裝...
    pip install -r requirements.txt
)
echo.

echo ========================================
echo 選擇操作:
echo ========================================
echo 1. 快速測試訓練 (20 epochs)
echo 2. 完整訓練 (所有配置)
echo 3. 僅訓練集成模型
echo 4. 生成預測
echo 5. 查看項目結構
echo 6. 退出
echo ========================================
echo.

set /p choice="請選擇 (1-6): "

if "%choice%"=="1" (
    echo.
    echo 開始快速測試訓練...
    python train_competition.py --mode quick
) else if "%choice%"=="2" (
    echo.
    echo 開始完整訓練（這將需要很長時間）...
    echo 確定要繼續嗎？
    pause
    python train_competition.py --mode all
) else if "%choice%"=="3" (
    echo.
    echo 開始訓練集成模型...
    python train_competition.py --mode ensemble
) else if "%choice%"=="4" (
    echo.
    echo 預測功能
    echo 用法: python predict.py --test-data TEST_FILE --models MODEL1 MODEL2 ...
    echo.
    echo 示例:
    echo python predict.py --test-data Data/test.npz --models Outputs/competition/affi/patchtst_v1/best_model.pt
) else if "%choice%"=="5" (
    echo.
    echo 項目結構:
    tree /F /A
) else if "%choice%"=="6" (
    echo.
    echo 再見！
    exit /b 0
) else (
    echo.
    echo 無效選擇！
)

echo.
echo ========================================
echo 操作完成！
echo ========================================
pause
