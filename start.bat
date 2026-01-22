@echo off
echo Activating virtual environment...
call venv\Scripts\activate


echo Setting environment variables...
rem HF_HOME override disabled - using default ~/.cache/huggingface for token compatibility
rem set HF_HOME=%~dp0cache
rem if not exist "%HF_HOME%" mkdir "%HF_HOME%"

echo Starting Pocket TTS API...
echo Please check the log below for the actual PORT (e.g. 8001).
python pocketapi.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Server crashed! Check the error message above.
    pause
    exit /b %ERRORLEVEL%
)

pause
