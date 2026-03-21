@echo off
REM Install dependencies for Ice Open TTS Proxy (Windows)

cd /d "%~dp0"

echo ==========================================
echo   Ice Open TTS Proxy - Installer
echo ==========================================
echo.

set VENV_DIR=..\venv

REM Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    py --version >nul 2>&1
    if %errorlevel% neq 0 (
        echo X Python not found. Install Python 3.8+
        pause
        exit /b 1
    )
    set PYTHON=py
) else (
    set PYTHON=python
)

%PYTHON% --version
echo.

REM Create venv if needed
if not exist "%VENV_DIR%" (
    echo Creating virtual environment...
    %PYTHON% -m venv "%VENV_DIR%"
    if %errorlevel% neq 0 (
        echo X Failed to create venv
        pause
        exit /b 1
    )
    echo + Virtual environment created
)

REM Activate venv
call "%VENV_DIR%\Scripts\activate.bat"

REM Upgrade pip
echo Upgrading pip...
pip install --upgrade pip -q

REM Install required packages
echo.
echo Installing required packages...
pip install -q requests

REM Install optional packages
echo.
echo Installing optional packages...
pip install -q flask 2>nul && echo   + Flask (API server) || echo   - Flask (API server) skipped
pip install -q simpleaudio 2>nul && echo   + SimpleAudio (audio) || echo   - SimpleAudio skipped  
pip install -q playsound 2>nul && echo   + Playsound (audio) || echo   - Playsound skipped

REM Test GUI support
echo.
echo Checking GUI support...
%PYTHON% -c "import tkinter; print('  + tkinter available (GUI supported)')" 2>nul || (
    echo   - tkinter not available - GUI disabled
)

echo.
echo ==========================================
echo   Installation complete!
echo ==========================================
echo.
echo Usage:
echo   GUI:  start_ice_gui.bat
echo   CLI:  start_ice_cli.bat
echo.
pause
