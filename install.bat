@echo off
setlocal
rem Always install into the project directory, wherever the script is run from.
pushd "%~dp0" || exit /b 1

echo Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo Failed to create the virtual environment. Install Python 3.10 or newer and add it to PATH.
    goto fail
)
echo Activating virtual environment...
call venv\Scripts\activate

ffmpeg -version > nul 2>&1
if errorlevel 1 (
    echo WARNING: ffmpeg could not be found. Install it and add it to PATH. It is required for format 'mp3' and speed adjustments.
)

echo Upgrading pip...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo Failed to upgrade pip.
    goto fail
)

set /p install_cuda=Do you want to install PyTorch with CUDA support? (y/n):
if /I "%install_cuda%" == "y" (
    echo Installing PyTorch with CUDA support...
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
    if errorlevel 1 (
        echo Failed to install PyTorch with CUDA support.
        goto fail
    )
) else (
    echo Proceeding with default PyTorch installation...
)

echo Installing dependencies...
python -m pip install -e .
if errorlevel 1 (
    echo Failed to install dependencies. Installation incomplete.
    goto fail
)
echo Installation complete.
pause
popd
exit /b 0

:fail
pause
popd
exit /b 1
