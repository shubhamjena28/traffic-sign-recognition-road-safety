@echo off
title Traffic Sign Recognition & Road Safety Monitor - GUI
echo =====================================================================
echo  Traffic Sign Recognition and Road Safety Monitor - Streamlit GUI
echo  PBL College Project Demonstration
echo =====================================================================
echo.

if exist ".venv\Scripts\activate.bat" (
    echo Activating Python virtual environment (.venv)...
    call .venv\Scripts\activate.bat
) else (
    echo [NOTE] Local .venv not detected. Using active environment...
)

echo.
echo Launching Streamlit GUI Application on http://localhost:8501 ...
echo Press Ctrl+C in this window to stop the server.
echo.

streamlit run app.py

pause
