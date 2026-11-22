#!/bin/bash
# Start both OpenClaw Voice servers
# Usage: ./run.sh
# Uses .venv311 (Python 3.11) for XTTS-v2 support

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Activate venv311 (Python 3.11 with Coqui TTS/XTTS-v2 support)
if [ -f ".venv311/bin/activate" ]; then
    source .venv311/bin/activate
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Load environment
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
fi

echo "Starting OpenClaw Voice servers..."
echo ""

# Start main server (port 8765)
echo "  [1] Main server on port ${OPENCLAW_PORT:-8765}"
PYTHONPATH=. python -m src.server.main &
MAIN_PID=$!

# Start Deepgram Mimic server (port 8766)
echo "  [2] Deepgram Mimic server on port ${PORT:-8766}"
PYTHONPATH=. python deepgram_mimic.py &
MIMIC_PID=$!

echo ""
echo "Both servers running. Press Ctrl+C to stop."

# Wait and handle cleanup
trap "kill $MAIN_PID $MIMIC_PID 2>/dev/null; exit" INT TERM
wait
