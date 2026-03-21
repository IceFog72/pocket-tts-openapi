#!/bin/bash

# Default port if not specified
DEFAULT_PORT=8005
SCRIPT_NAME="pocketapi.py"

# Read port from config.ini
if [ -f "config.ini" ]; then
    CONFIG_PORT=$(grep -E '^server_port\s*=' config.ini | head -1 | sed 's/.*=\s*//' | tr -d '[:space:]')
    if [ -n "$CONFIG_PORT" ] && [ "$CONFIG_PORT" -eq "$CONFIG_PORT" ] 2>/dev/null; then
        DEFAULT_PORT=$CONFIG_PORT
    fi
fi

# Check if port is in use and get PID + process name
check_port() {
    local port=$1
    local pid=""
    local process_name=""
    
    if command -v lsof &> /dev/null; then
        pid=$(lsof -i :$port -t 2>/dev/null | head -1)
        if [ -n "$pid" ]; then
            process_name=$(ps -p $pid -o comm= 2>/dev/null)
        fi
    elif command -v ss &> /dev/null; then
        local info=$(ss -tlnp 2>/dev/null | grep ":$port " | head -1)
        pid=$(echo "$info" | grep -oP 'pid=\K[0-9]+' | head -1)
        if [ -n "$pid" ]; then
            process_name=$(ps -p $pid -o comm= 2>/dev/null)
        fi
    elif command -v netstat &> /dev/null; then
        local info=$(netstat -tlnp 2>/dev/null | grep ":$port " | head -1)
        pid=$(echo "$info" | awk '{print $7}' | cut -d'/' -f1)
        if [ -n "$pid" ]; then
            process_name=$(ps -p $pid -o comm= 2>/dev/null)
        fi
    fi
    
    echo "$pid:$process_name"
}

# Check if process is our TTS server
is_our_server() {
    local process_name=$1
    local pid=$2
    
    # Check by process name
    if [[ "$process_name" == *"python"* ]] || [[ "$process_name" == *"python3"* ]]; then
        # Check command line for our script
        if command -v ps &> /dev/null && [ -n "$pid" ]; then
            local cmdline=$(ps -p $pid -o args= 2>/dev/null)
            if [[ "$cmdline" == *"$SCRIPT_NAME"* ]]; then
                return 0  # It's our server
            fi
        fi
    fi
    
    # Also check for "pocket" in process name
    if [[ "$process_name" == *"pocket"* ]]; then
        return 0
    fi
    
    return 1  # Not our server
}

# Get port info
PORT_INFO=$(check_port $DEFAULT_PORT)
PID=$(echo "$PORT_INFO" | cut -d':' -f1)
PROCESS_NAME=$(echo "$PORT_INFO" | cut -d':' -f2)

if [ -n "$PID" ]; then
    # Port is in use - check if it's our TTS server
    if is_our_server "$PROCESS_NAME" "$PID"; then
        # It's our TTS server - show all 3 options
        echo ""
        echo "============================================================"
        echo "  Pocket TTS already running on port $DEFAULT_PORT (PID: $PID)"
        echo "============================================================"
        echo ""
        echo "What would you like to do?"
        echo ""
        echo "  1) Kill and restart server"
        echo "  2) Start on a different port"
        echo "  3) Exit"
        echo ""
        read -p "Enter choice [1-3]: " choice
        
        case $choice in
            1)
                echo ""
                echo "Killing existing server (PID: $PID)..."
                kill $PID 2>/dev/null
                sleep 2
                # Force kill if still running
                if [ -n "$(check_port $DEFAULT_PORT | cut -d':' -f1)" ]; then
                    echo "Force killing..."
                    kill -9 $PID 2>/dev/null
                    sleep 1
                fi
                echo "Restarting server..."
                ;;
            2)
                echo ""
                read -p "Enter new port number: " NEW_PORT
                if [ -n "$NEW_PORT" ] && [ "$NEW_PORT" -eq "$NEW_PORT" 2>/dev/null ]; then
                    NEW_INFO=$(check_port $NEW_PORT)
                    NEW_PID=$(echo "$NEW_INFO" | cut -d':' -f1)
                    if [ -n "$NEW_PID" ]; then
                        echo "Error: Port $NEW_PORT is also in use (PID: $NEW_PID)"
                        exit 1
                    fi
                    export OVERRIDE_PORT=$NEW_PORT
                    echo "Starting server on port $NEW_PORT..."
                else
                    echo "Invalid port number"
                    exit 1
                fi
                ;;
            3)
                echo "Exiting."
                exit 0
                ;;
            *)
                echo "Invalid choice. Exiting."
                exit 1
                ;;
        esac
    else
        # It's another application - only show 2 options
        echo ""
        echo "============================================================"
        echo "  Port $DEFAULT_PORT is in use by another application"
        echo "  Process: $PROCESS_NAME (PID: $PID)"
        echo "============================================================"
        echo ""
        echo "What would you like to do?"
        echo ""
        echo "  1) Start on a different port"
        echo "  2) Exit"
        echo ""
        read -p "Enter choice [1-2]: " choice
        
        case $choice in
            1)
                echo ""
                read -p "Enter new port number: " NEW_PORT
                if [ -n "$NEW_PORT" ] && [ "$NEW_PORT" -eq "$NEW_PORT" 2>/dev/null ]; then
                    NEW_INFO=$(check_port $NEW_PORT)
                    NEW_PID=$(echo "$NEW_INFO" | cut -d':' -f1)
                    if [ -n "$NEW_PID" ]; then
                        echo "Error: Port $NEW_PORT is also in use (PID: $NEW_PID)"
                        exit 1
                    fi
                    export OVERRIDE_PORT=$NEW_PORT
                    echo "Starting server on port $NEW_PORT..."
                else
                    echo "Invalid port number"
                    exit 1
                fi
                ;;
            2)
                echo "Exiting."
                exit 0
                ;;
            *)
                echo "Invalid choice. Exiting."
                exit 1
                ;;
        esac
    fi
fi

echo ""
echo "Activating virtual environment..."
source venv/bin/activate

export OVERRIDE_PORT=$DEFAULT_PORT

echo "Starting Pocket TTS API..."
echo "Please check the log below for the actual PORT."
python pocketapi.py

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "Server crashed! Check the error message above."
    read -p "Press Enter to exit..."
    exit $EXIT_CODE
fi
