@echo off
REM Quick launcher for the SEM/TEM particle analysis Gradio app (Windows).
REM Activates a conda environment and starts the app from the repo root.

REM Set console to UTF-8 to handle emoji and special characters
chcp 65001 > nul 2>&1

REM Run from the repo root whatever directory this was launched from, so
REM "python -m sem_analysis_app" can find the package.
cd /d "%~dp0"

echo ========================================
echo  SEM Particle Analysis - Gradio App
echo ========================================
echo.

REM The conda environment to use (README.md, "Segmentation tool").  Override
REM with   set SEM_ENV=my-env   before running, or edit the default here.
if "%SEM_ENV%"=="" set SEM_ENV=SEM_analysis
if "%CONDA_ROOT%"=="" set CONDA_ROOT=%USERPROFILE%\anaconda3
if not exist "%CONDA_ROOT%" if exist "%USERPROFILE%\miniconda3" set CONDA_ROOT=%USERPROFILE%\miniconda3

if not exist "%CONDA_ROOT%\envs\%SEM_ENV%" (
    echo ERROR: conda environment "%SEM_ENV%" not found under %CONDA_ROOT%\envs
    echo Create it ^(see README.md^), or set SEM_ENV / CONDA_ROOT before running this script.
    pause
    exit /b 1
)

echo Activating %SEM_ENV% environment...
call "%CONDA_ROOT%\Scripts\activate.bat" %SEM_ENV%

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

python -m sem_analysis_app

pause
