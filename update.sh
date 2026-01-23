#!/bin/bash

# Change to the script's directory
cd "$(dirname "$0")"

echo ""
echo "========================================"
echo "  Updating Pocket TTS OpenAPI"
echo "========================================"
echo ""

# Check for git
if ! command -v git &> /dev/null; then
    echo -e "\e[91mGit is not installed on this system.\e[0m"
    echo "Please install git using your package manager."
    exit 1
fi

# Check for .git directory
if [ ! -d ".git" ]; then
    echo -e "\e[91mNot running from a Git repository. Reinstall using git clone to get updates.\e[0m"
    exit 1
fi

echo "Pulling latest changes..."
git pull --rebase --autostash

if [ $? -ne 0 ]; then
    echo ""
    echo -e "\e[91mThere were errors while updating.\e[0m"
    echo "Please check for local conflicts or network issues."
    echo ""
    exit 1
fi

echo ""
echo "Updating dependencies..."
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    python3 -m pip install --upgrade pip
    pip install -e .
else
    echo -e "\e[93mVirtual environment (venv) not found. Skipping dependency update.\e[97m"
    echo "Run ./install.sh first if this is a new installation."
fi

echo ""
echo -e "\e[92m[SUCCESS] Update complete.\e[0m"
echo ""
