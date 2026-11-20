#!/bin/bash
set -e

# Always install into the project directory, wherever the script is run from.
cd "$(dirname "$0")"

echo "Creating virtual environment..."
python3 -m venv venv

echo "Activating virtual environment..."
source venv/bin/activate

if ! command -v ffmpeg &> /dev/null; then
    echo "WARNING: ffmpeg could not be found. Please install it via your system package manager (e.g., sudo pacman -S ffmpeg, sudo apt install ffmpeg), as it is required for format 'mp3' and speed adjustments."
fi

echo "Upgrading pip..."
python -m pip install --upgrade pip

# Install torch before the project so pip keeps this build instead of PyPI's
# default Linux wheel, which bundles several GB of CUDA libraries.
echo "Select the PyTorch backend:"
echo "  1) CPU only (default)"
echo "  2) NVIDIA GPU (CUDA 12.8)"
echo "  3) AMD GPU (ROCm 10.0)"
read -p "Choice [1-3]: " torch_backend
case "$torch_backend" in
    2) torch_index=cu128 ;;
    3) torch_index=rocm10.0 ;;
    *) torch_index=cpu ;;
esac
echo "Installing PyTorch ($torch_index)..."
pip install torch --index-url "https://download.pytorch.org/whl/$torch_index"

echo "Installing dependencies..."
pip install -e .

echo "Installation complete!"
