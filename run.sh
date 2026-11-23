#!/bin/bash
# OpenClaw Voice - Unified Server Manager
# Manages both the main server (port 8765) and OpenAI Mimic (port 8768)
# Usage: ./run.sh {start|stop|status|edit}

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Venv
VENV_DIR="$SCRIPT_DIR/.venv311"
if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
elif [ -f "$SCRIPT_DIR/.venv/bin/activate" ]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
fi

# PID files
MAIN_PID_FILE="/tmp/openclaw_voice_main.pid"
MIMIC_PID_FILE="/tmp/openai_mimic.pid"

# Ports
MAIN_PORT="${OPENCLAW_PORT:-8765}"
MIMIC_PORT="${PORT:-8768}"

# Load environment
if [ -f "$SCRIPT_DIR/.env" ]; then
    export $(grep -v '^#' "$SCRIPT_DIR/.env" | xargs)
fi

# ===================
# Helper functions
# ===================

_is_running() {
    local pid_file="$1"
    if [ -f "$pid_file" ]; then
        local pid
        pid=$(cat "$pid_file" 2>/dev/null)
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            echo "$pid"
            return 0
        else
            rm -f "$pid_file"
            return 1
        fi
    fi
    return 1
}

_start_main() {
    if _is_running "$MAIN_PID_FILE" >/dev/null 2>&1; then
        echo "  Main server already running (PID: $(_is_running $MAIN_PID_FILE))"
        return
    fi
    echo "  Starting main server on port $MAIN_PORT..."
    nohup env PYTHONPATH="$SCRIPT_DIR" python -m src.server.main > /tmp/openclaw_voice_main.log 2>&1 &
    local pid=$!
    echo $pid > "$MAIN_PID_FILE"
    echo "  Main server started (PID: $pid)"
}

_start_mimic() {
    if _is_running "$MIMIC_PID_FILE" >/dev/null 2>&1; then
        echo "  OpenAI Mimic already running (PID: $(_is_running $MIMIC_PID_FILE))"
        return
    fi
    echo "  Starting OpenAI Mimic on port $MIMIC_PORT..."
    nohup python openai_mimic.py start > /tmp/openai_mimic_start.log 2>&1 &
    # Wait for PID file to be created by the mimic's own start command
    for i in $(seq 1 30); do
        if [ -f "$MIMIC_PID_FILE" ]; then
            break
        fi
        sleep 1
    done
    if [ -f "$MIMIC_PID_FILE" ]; then
        echo "  OpenAI Mimic started (PID: $(cat $MIMIC_PID_FILE))"
    else
        echo "  OpenAI Mimic starting (check /tmp/openai_mimic_start.log)"
    fi
}

_stop_main() {
    local pid
    pid=$(_is_running "$MAIN_PID_FILE" 2>/dev/null)
    if [ -n "$pid" ]; then
        kill "$pid" 2>/dev/null
        rm -f "$MAIN_PID_FILE"
        echo "  Main server stopped (PID: $pid)"
    else
        echo "  Main server not running"
    fi
}

_stop_mimic() {
    local pid
    pid=$(_is_running "$MIMIC_PID_FILE" 2>/dev/null)
    if [ -n "$pid" ]; then
        kill "$pid" 2>/dev/null
        rm -f "$MIMIC_PID_FILE"
        echo "  OpenAI Mimic stopped (PID: $pid)"
    else
        echo "  OpenAI Mimic not running"
    fi
}

_status_main() {
    local pid
    pid=$(_is_running "$MAIN_PID_FILE" 2>/dev/null)
    if [ -n "$pid" ]; then
        echo "  Main server: RUNNING (PID: $pid, port $MAIN_PORT)"
        curl -s "http://localhost:$MAIN_PORT/health" 2>/dev/null | python3 -m json.tool 2>/dev/null || echo "    (unreachable)"
    else
        echo "  Main server: NOT running"
    fi
}

_status_mimic() {
    local pid
    pid=$(_is_running "$MIMIC_PID_FILE" 2>/dev/null)
    if [ -n "$pid" ]; then
        echo "  OpenAI Mimic: RUNNING (PID: $pid, port $MIMIC_PORT)"
        curl -s "http://localhost:$MIMIC_PORT/health" 2>/dev/null | python3 -m json.tool 2>/dev/null || echo "    (unreachable)"
    else
        echo "  OpenAI Mimic: NOT running"
    fi
}

cmd_start() {
    echo "Starting OpenClaw Voice servers..."
    echo ""
    _start_main
    _start_mimic
    echo ""
    echo "Both servers starting. Check status with: ./run.sh status"
}

cmd_stop() {
    echo "Stopping OpenClaw Voice servers..."
    echo ""
    _stop_main
    _stop_mimic
    echo ""
    echo "All servers stopped."
}

cmd_status() {
    echo "OpenClaw Voice Server Status:"
    echo ""
    _status_main
    echo ""
    _status_mimic
}

cmd_edit() {
    local config_file="$SCRIPT_DIR/.env"
    echo "Configuration file: $config_file"
    echo ""
    if [ -f "$config_file" ]; then
        echo "Current .env contents:"
        echo "----------------------------------------"
        cat "$config_file"
        echo "----------------------------------------"
    else
        echo "No .env file found. Creating from template..."
        cat > "$config_file" << 'ENVEOF'
# OpenClaw Voice Configuration
# Using local LLM (port 1919) as AI backend
# Using local XTTS-v2 for TTS (no ElevenLabs needed)

# AI Backend (local LLM)
OPENCLAW_OPENAI_API_KEY=not-needed
OPENCLAW_BACKEND_URL=http://localhost:1919/v1
OPENCLAW_BACKEND_MODEL=Qwen3.6-35B-A3B-UNC-NVFP4-ftw

# Server settings
OPENCLAW_HOST=0.0.0.0
OPENCLAW_PORT=8765
OPENCLAW_STT_MODEL=base
OPENCLAW_STT_DEVICE=auto
OPENCLAW_REQUIRE_AUTH=false

# OpenAI Mimic settings
PORT=8768
HOST=0.0.0.0
WHISPER_MODEL=base
WHISPER_DEVICE=auto
ENVEOF
        echo "Created .env template"
    fi

    echo ""
    echo "To edit, modify the .env file directly:"
    echo "  nano $config_file"
    echo "  vim $config_file"
    echo ""
    echo "Available settings:"
    echo "  OPENCLAW_OPENAI_API_KEY  - API key for LLM (use 'not-needed' for local)"
    echo "  OPENCLAW_BACKEND_URL     - Local LLM URL (default: http://localhost:1919/v1)"
    echo "  OPENCLAW_BACKEND_MODEL   - Model name"
    echo "  OPENCLAW_PORT            - Main server port (default: 8765)"
    echo "  OPENCLAW_STT_MODEL       - Whisper model (tiny, base, small, medium, large-v3-turbo)"
    echo "  OPENCLAW_STT_DEVICE      - Device (auto, cpu, cuda, mps)"
    echo "  OPENCLAW_REQUIRE_AUTH    - Require API keys (true/false)"
    echo "  PORT                     - OpenAI Mimic port (default: 8768)"
    echo "  WHISPER_MODEL            - Whisper model for mimic (default: base)"
    echo "  VOICE_SAMPLE             - Path to voice sample WAV for XTTS cloning"
    echo ""
    echo "Note: Restart servers after editing .env"
}

# ===================
# Main
# ===================

if [ $# -eq 0 ]; then
    cmd_start
else
    case "$1" in
        start)
            cmd_start
            ;;
        stop)
            cmd_stop
            ;;
        status)
            cmd_status
            ;;
        edit)
            cmd_edit
            ;;
        *)
            echo "Usage: $0 {start|stop|status|edit}"
            exit 1
            ;;
    esac
fi