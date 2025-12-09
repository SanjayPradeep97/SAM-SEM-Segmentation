@echo off
REM CNT Particle Analysis App Launcher (Windows)

echo ============================================
echo CNT Particle Analysis Application
echo ============================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

echo Checking dependencies...
python -c "import gradio" 2>nul
if errorlevel 1 (
    echo Dependencies not found. Installing...
    pip install -r requirements.txt
)

echo.
echo Starting Gradio app...
echo Open your browser to: http://127.0.0.1:7860
echo.
echo Press Ctrl+C to stop the application
echo.

REM Launch the app
python app.py

pause
