@echo off
cd /d "%~dp0"
title Traffic Sign Recognition - Streamlit GUI

echo =====================================================================
echo  Traffic Sign Recognition and Road Safety Monitor - Streamlit GUI
echo  PBL College Project Demonstration
echo =====================================================================
echo.

if exist ".venv\Scripts\python.exe" (
    echo [INFO] Found local virtual environment.
    echo [INFO] Starting Streamlit dashboard on http://localhost:8501 ...
    echo [INFO] Press Ctrl+C in this window to stop the server.
    echo.
    ".venv\Scripts\python.exe" -m streamlit run app.py
) else (
    echo [INFO] Using system Python.
    echo [INFO] Starting Streamlit dashboard on http://localhost:8501 ...
    echo.
    python -m streamlit run app.py
)

echo.
echo =====================================================================
echo  Streamlit session ended.
echo =====================================================================
pause
