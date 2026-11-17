@echo off
cd /d "%~dp0"
title Traffic Sign Recognition & Road Safety Monitor - GUI
echo =====================================================================
echo  Traffic Sign Recognition and Road Safety Monitor - Streamlit GUI
echo  PBL College Project Demonstration
echo =====================================================================
echo.

if exist ".venv\Scripts\activate.bat" (
    echo Activating Python virtual environment (.venv)...
    call .venv\Scripts\activate.bat
    echo Launching Streamlit dashboard...
    python -m streamlit run app.py
) else (
    echo Starting Streamlit dashboard...
    python -m streamlit run app.py
)

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [NOTE] If you see an error above, press any key to close.
)
pause
