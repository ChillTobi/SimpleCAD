@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo [error] .venv not found. Run: python build.py --venv
    exit /b 1
)
call .venv\Scripts\activate.bat
python SimpleCAD.py
