@echo off
pushd %~dp0
echo.
echo ========================================
echo   Updating Pocket TTS OpenAPI
echo ========================================
echo.

git --version > nul 2>&1
if %errorlevel% neq 0 (
    echo [91mGit is not installed on this system.[0m
    echo Install it from https://git-scm.com/downloads
    goto end
) else (
    if not exist .git (
        echo [91mNot running from a Git repository. Reinstall using git clone to get updates.[0m
        goto end
    )
    
    echo Pulling latest changes...
    call git pull --rebase --autostash
    if %errorlevel% neq 0 (
        echo.
        echo [91mThere were errors while updating.[0m
        echo Please check for local conflicts or network issues.
        goto end
    )
)

echo.
echo Updating dependencies...
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
    python -m pip install --upgrade pip
    pip install -e .
) else (
    echo [93mVirtual environment (venv) not found. Skipping dependency update.[0m
    echo Run install.bat first if this is a new installation.
)

echo.
echo [92m[SUCCESS] Update complete.[0m
echo.

:end
pause
popd
