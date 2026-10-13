@echo off
setlocal
pushd "%~dp0" || exit /b 1
echo.
echo ========================================
echo   Updating Pocket TTS OpenAPI
echo ========================================
echo.

git --version > nul 2>&1
if errorlevel 1 (
    echo Git is not installed on this system.
    echo Install it from https://git-scm.com/downloads
    goto fail
)
if not exist .git (
    echo Not running from a Git repository. Reinstall using git clone to get updates.
    goto fail
)
if not exist venv\Scripts\python.exe (
    echo Virtual environment ^(venv^) not found. Run install.bat first.
    goto fail
)

echo Pulling latest changes...
call git pull --rebase --autostash
if errorlevel 1 (
    echo There were errors while updating.
    echo Please check for local conflicts or network issues.
    goto fail
)

echo.
echo Updating dependencies...
venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 (
    echo Failed to update pip. Update incomplete.
    goto fail
)
venv\Scripts\python.exe -m pip install -e .
if errorlevel 1 (
    echo Failed to install dependencies. Update incomplete.
    goto fail
)

echo.
echo [SUCCESS] Update complete.
echo.
pause
popd
exit /b 0

:fail
pause
popd
exit /b 1
