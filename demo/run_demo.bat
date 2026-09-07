@echo off
REM Launch the click-to-classify demo (Windows).
REM   1. activates the cnt-vfm conda env if it is not already active
REM   2. downloads the SAM ViT-B checkpoint (358 MB) into ..\sam_weights once
REM   3. starts the Gradio page at http://127.0.0.1:7861
setlocal
cd /d "%~dp0"
set "PYTHONIOENCODING=utf-8"
set "KMP_DUPLICATE_LIB_OK=TRUE"

if not "%CONDA_DEFAULT_ENV%"=="cnt-vfm" (
  if "%CONDA_ROOT%"=="" set "CONDA_ROOT=%USERPROFILE%\anaconda3"
  if not exist "%CONDA_ROOT%" if exist "%USERPROFILE%\miniconda3" set "CONDA_ROOT=%USERPROFILE%\miniconda3"
  call "%CONDA_ROOT%\Scripts\activate.bat" "%CONDA_ROOT%" 2>nul
  call conda activate cnt-vfm || (
    echo could not activate the conda env cnt-vfm.
    echo create it first:   conda env create -f ..\environment.yml   ^(or environment-cpu.yml^)
    pause & exit /b 1
  )
)

if not exist "..\sam_weights\sam_vit_b_01ec64.pth" (
  echo downloading the SAM ViT-B checkpoint ^(358 MB, once^) ...
  python ..\download_sam_weights.py --model vit_b || (echo download failed & pause & exit /b 1)
)

python demo_app.py %*
pause
