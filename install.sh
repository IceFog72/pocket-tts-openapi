#!/bin/bash
set -e

echo "Creating virtual environment..."
python3 -m venv venv

echo "Activating virtual environment..."
source venv/bin/activate

if ! command -v ffmpeg &> /dev/null; then
    echo "WARNING: ffmpeg could not be found. Please install it via your system package manager (e.g., sudo pacman -S ffmpeg, sudo apt install ffmpeg), as it is required for format 'mp3' and speed adjustments."
fi

echo "Upgrading pip..."
python -m pip install --upgrade pip

read -p "Do you want to install PyTorch with CUDA support? (y/n): " install_cuda
if [[ "$install_cuda" =~ ^[Yy]$ ]]; then
    echo "Installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
else
    echo "Proceeding with default PyTorch installation..."
fi

echo "Installing dependencies..."
pip install -e .

echo "Installation complete!"
