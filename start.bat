@echo off
setlocal
rem config.ini and data directories are resolved from the working directory.
pushd "%~dp0" || exit /b 1

if not exist venv\Scripts\python.exe (
    echo Virtual environment ^(venv^) not found. Run install.bat first.
    pause
    popd
    exit /b 1
)

echo Activating virtual environment...
call venv\Scripts\activate

echo Setting environment variables...
rem HF_HOME override disabled - using default ~/.cache/huggingface for token compatibility
rem set HF_HOME=%~dp0cache
rem if not exist "%HF_HOME%" mkdir "%HF_HOME%"

echo Starting Pocket TTS API...
echo Please check the log below for the actual PORT (e.g. 8005).
python pocketapi.py
set EXIT_CODE=%ERRORLEVEL%
if not "%EXIT_CODE%" == "0" (
    echo.
    echo Server crashed! Check the error message above.
    pause
    popd
    exit /b %EXIT_CODE%
)

pause
popd
