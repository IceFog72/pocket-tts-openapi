#!/bin/bash
set -e

echo "Creating virtual environment..."
python3 -m venv venv

echo "Activating virtual environment..."
source venv/bin/activate

echo "Upgrading pip..."
python -m pip install --upgrade pip

read -p "Do you want to install PyTorch with CUDA support? (y/n): " install_cuda
if [[ "$install_cuda" =~ ^[Yy]$ ]]; then
    echo "Installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    echo "Proceeding with default PyTorch installation..."
fi

echo "Installing dependencies..."
pip install -e .

echo "Installation complete!"
