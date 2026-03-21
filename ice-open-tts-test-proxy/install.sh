#!/bin/bash
# Install dependencies for Ice Open TTS Proxy (Linux/macOS)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "  Ice Open TTS Proxy - Installer"
echo "=========================================="
echo ""

VENV_DIR="../venv"
PYTHON="python3"

# Check Python
if ! command -v $PYTHON &> /dev/null; then
    PYTHON="python"
    if ! command -v $PYTHON &> /dev/null; then
        echo "✗ Python not found. Install Python 3.8+"
        exit 1
    fi
fi

echo "✓ Python: $($PYTHON --version)"

# Detect package manager (Linux only)
detect_pkg_manager() {
    if command -v pacman &> /dev/null; then
        echo "pacman"  # Arch, Manjaro, EndeavourOS
    elif command -v yay &> /dev/null; then
        echo "yay"    # Arch AUR helper
    elif command -v paru &> /dev/null; then
        echo "paru"   # Arch AUR helper
    elif command -v apt-get &> /dev/null; then
        echo "apt"    # Debian, Ubuntu, Mint
    elif command -v dnf &> /dev/null; then
        echo "dnf"    # Fedora
    elif command -v zypper &> /dev/null; then
        echo "zypper" # openSUSE
    elif command -v emerge &> /dev/null; then
        echo "emerge" # Gentoo
    elif command -v apk &> /dev/null; then
        echo "apk"    # Alpine
    else
        echo "unknown"
    fi
}

install_system_pkg() {
    local pkg="$1"
    local manager=$(detect_pkg_manager)
    
    case $manager in
        pacman)
            sudo pacman -S --noconfirm "$pkg" 2>/dev/null
            ;;
        yay)
            yay -S --noconfirm "$pkg" 2>/dev/null
            ;;
        paru)
            paru -S --noconfirm "$pkg" 2>/dev/null
            ;;
        apt)
            sudo apt-get install -y "$pkg" 2>/dev/null
            ;;
        dnf)
            sudo dnf install -y "$pkg" 2>/dev/null
            ;;
        zypper)
            sudo zypper install -y "$pkg" 2>/dev/null
            ;;
        emerge)
            sudo emerge "$pkg" 2>/dev/null
            ;;
        apk)
            sudo apk add "$pkg" 2>/dev/null
            ;;
        *)
            return 1
            ;;
    esac
}

get_tkinter_pkg() {
    local manager=$(detect_pkg_manager)
    case $manager in
        pacman|yay|paru) echo "tk" ;;
        apt)              echo "python3-tk" ;;
        dnf)              echo "python3-tkinter" ;;
        zypper)           echo "python3-tk" ;;
        emerge)           echo "dev-lang/tk" ;;
        apk)              echo "tk" ;;
        *)                echo "python3-tk" ;;
    esac
}

get_audio_pkg() {
    local manager=$(detect_pkg_manager)
    case $manager in
        pacman|yay|paru) echo "pulseaudio-alsa" ;;
        apt)              echo "pulseaudio" ;;
        dnf)              echo "pulseaudio" ;;
        zypper)           echo "pulseaudio" ;;
        *)                echo "pulseaudio" ;;
    esac
}

PKG_MANAGER=$(detect_pkg_manager)
echo "📦 Package manager: $PKG_MANAGER"

# Create venv if needed
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    $PYTHON -m venv "$VENV_DIR"
    if [ $? -ne 0 ]; then
        echo "✗ Failed to create venv"
        exit 1
    fi
    echo "✓ Virtual environment created"
fi

# Activate venv
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip -q

# Install required packages
echo ""
echo "Installing Python packages..."
pip install -q requests
echo "  ✓ requests"

pip install -q flask 2>/dev/null && echo "  ✓ Flask" || echo "  ⚠ Flask failed"
pip install -q simpleaudio 2>/dev/null && echo "  ✓ SimpleAudio" || echo "  ⚠ SimpleAudio failed"

# Test audio backend
echo ""
echo "Testing audio backend..."
$PYTHON -c "
try:
    import simpleaudio
    print('  ✓ SimpleAudio available')
except ImportError:
    if __import__('platform').system() == 'Linux':
        import subprocess
        for cmd in ['paplay', 'aplay', 'ffplay']:
            try:
                subprocess.run(['which', cmd], check=True, capture_output=True)
                print(f'  ✓ Found {cmd}')
                break
            except:
                pass
        else:
            print('  ⚠ No audio backend')
    else:
        print('  ⚠ No audio backend')
"

# Check and offer to install tkinter
echo ""
echo "Checking GUI support..."
$PYTHON -c "import tkinter; print('  ✓ tkinter available (GUI supported)')" 2>/dev/null

if [ $? -ne 0 ]; then
    echo "  ⚠ tkinter not available (GUI disabled)"
    
    if [ "$PKG_MANAGER" != "unknown" ]; then
        TK_PKG=$(get_tkinter_pkg)
        echo ""
        read -p "Install tkinter ($TK_PKG)? [Y/n] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
            echo "Installing $TK_PKG..."
            install_system_pkg "$TK_PKG"
            
            # Verify
            $PYTHON -c "import tkinter" 2>/dev/null && echo "  ✓ tkinter installed!" || echo "  ✗ tkinter install failed"
        fi
    else
        echo ""
        echo "  Install tkinter manually for your distro"
    fi
fi

echo ""
echo "=========================================="
echo "  Installation complete!"
echo "=========================================="
echo ""
echo "Usage:"
echo "  GUI:  ./start_ice_gui.sh"
echo "  CLI:  ./start_ice_cli.sh [port]"
echo ""
