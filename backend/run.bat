@echo off
title Shelfy.ai Backend Server (Port 8000)
cd /d "%~dp0"
if exist "venv\Scripts\python.exe" (
    echo [OK] Launching uvicorn with venv...
    venv\Scripts\python.exe -m uvicorn main:app --reload --port 8000
) else (
    echo [ERROR] Virtual environment not found in backend\venv
    pause
)
