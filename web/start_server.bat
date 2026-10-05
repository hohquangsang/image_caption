@echo off
echo ========================================
echo  VisioCaption AI - Khoi dong server
echo ========================================
echo.

cd /d %~dp0

:: Kiem tra Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [LOI] Python chua duoc cai dat!
    echo Vui long cai Python 3.8+ truoc
    pause
    exit /b 1
)

:: Kiem tra pip packages
echo [1/3] Kiem tra thu vien...
pip install flask flask-cors pillow >nul 2>&1
echo     OK!

echo [2/3] Kiem tra model files...
if not exist "..\caption_model_v2.keras" (
    echo [CANH BAO] Khong tim thay caption_model_v2.keras
    echo           Dat file model vao thu muc Image_Caption_Final/
)
if not exist "..\vocab_v2.pkl" (
    echo [CANH BAO] Khong tim thay vocab_v2.pkl
)
echo     OK!

echo [3/3] Khoi dong Flask server...
echo.
echo ========================================
echo  SERVER DANG CHAY tai: http://localhost:5000
echo  Nhan Ctrl+C de dung server
echo ========================================
echo.

python app.py

pause
