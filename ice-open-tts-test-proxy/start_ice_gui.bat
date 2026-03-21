@echo off
cd /d "%~dp0"
if exist "..\venv\Scripts\activate.bat" call ..\venv\Scripts\activate.bat
if "%~1"=="" (
    python ice_open_tts_proxy.py
) else (
    python ice_open_tts_proxy.py --port %1
)
pause
