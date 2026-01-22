#!/bin/bash

echo "Activating virtual environment..."
source venv/bin/activate

echo "Starting Pocket TTS API..."
echo "Please check the log below for the actual PORT (e.g. 8001)."
python pocketapi.py

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "Server crashed! Check the error message above."
    read -p "Press Enter to exit..."
    exit $EXIT_CODE
fi
