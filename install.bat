@echo off
echo Creating virtual environment...
python -m venv venv
echo Activating virtual environment...
call venv\Scripts\activate
echo Upgrading pip...
python -m pip install --upgrade pip

set /p install_cuda=Do you want to install PyTorch with CUDA support? (y/n): 
if /I "%install_cuda%" == "y" (
    echo Installing PyTorch with CUDA support...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
) else (
    echo Proceeding with default PyTorch installation...
)

echo Installing dependencies...
pip install -e .
echo Installation complete.
pause
