@echo off
REM Quick launcher for SEM Particle Analysis Gradio App
REM This script activates the conda environment and runs the application

REM Set console to UTF-8 to handle emoji and special characters
chcp 65001 > nul 2>&1

echo ========================================
echo  SEM Particle Analysis - Gradio App
echo ========================================
echo.

echo Activating SEM_analysis environment...
call C:\Users\sanja\anaconda3\Scripts\activate.bat SEM_analysis

echo.
echo Starting Gradio application...
echo The app will open at: http://127.0.0.1:7860
echo.
echo Press Ctrl+C to stop the server
echo ========================================
echo.

REM Set Python to use UTF-8 encoding
set PYTHONIOENCODING=utf-8

REM Fix OpenMP library conflict (Intel MKL + PyTorch)
set KMP_DUPLICATE_LIB_OK=TRUE

python sem_analysis_app\sem_analysis_app.py

pause
