@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>&1
if errorlevel 1 (
    echo Python launcher not found. Install Python 3.11 64-bit from https://www.python.org/downloads/
    echo Make sure "Add Python to PATH" is enabled.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating Python environment...
    py -3.11 -m venv .venv
    if errorlevel 1 goto fail
)

echo Installing desktop dependencies...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements-desktop.txt
if errorlevel 1 goto fail

echo Starting EmotionLens desktop app...
".venv\Scripts\python.exe" desktop_app.py
if errorlevel 1 goto fail
exit /b 0

:fail
echo.
echo Startup failed. Review the error above.
pause
exit /b 1
