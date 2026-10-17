#!/bin/bash

# Apourvoir Optimized Launcher
# Main launch script for Apourvoir with automatic hardware optimization
#
# Usage:
#   ./scripts/launcher/launch.sh [options]
#
# Options:
#   --model MODEL           Model to load (default: mistralai/Mixtral-8x7B-v0.1)
#   --preset PRESET         Optimization preset (balanced|speed|quality|memory|latency)
#   --no-api               Disable API server
#   --no-ui                Disable web UI
#   --api-host HOST        API server host (default: 0.0.0.0)
#   --api-port PORT        API server port (default: 8080)
#   --ui-port PORT         Web UI port (default: 8000)
#   --config FILE          Load configuration from file
#   --save-config FILE     Save configuration to file
#   --log-level LEVEL      Log level (DEBUG|INFO|WARNING|ERROR)
#   --info                 Show hardware info and exit
#   --validate             Validate configuration and exit
#   --help                Show this help message

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Add project directory to PYTHONPATH
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

# Default values
MODEL="mistralai/Mixtral-8x7B-v0.1"
PRESET="balanced"
ENABLE_API=true
ENABLE_UI=true
API_HOST="0.0.0.0"
API_PORT=8080
UI_PORT=8000
CONFIG_FILE=""
SAVE_CONFIG=""
LOG_LEVEL="INFO"
INFO_MODE=false
VALIDATE_MODE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --preset)
            PRESET="$2"
            shift 2
            ;;
        --no-api)
            ENABLE_API=false
            shift
            ;;
        --no-ui)
            ENABLE_UI=false
            shift
            ;;
        --api-host)
            API_HOST="$2"
            shift 2
            ;;
        --api-port)
            API_PORT="$2"
            shift 2
            ;;
        --ui-port)
            UI_PORT="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --save-config)
            SAVE_CONFIG="$2"
            shift 2
            ;;
        --log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        --info)
            INFO_MODE=true
            shift
            ;;
        --validate)
            VALIDATE_MODE=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --model MODEL           Model to load (default: $MODEL)"
            echo "  --preset PRESET         Optimization preset (balanced|speed|quality|memory|latency)"
            echo "  --no-api               Disable API server"
            echo "  --no-ui                Disable web UI"
            echo "  --api-host HOST        API server host (default: $API_HOST)"
            echo "  --api-port PORT        API server port (default: $API_PORT)"
            echo "  --ui-port PORT         Web UI port (default: $UI_PORT)"
            echo "  --config FILE          Load configuration from file"
            echo "  --save-config FILE     Save configuration to file"
            echo "  --log-level LEVEL      Log level (DEBUG|INFO|WARNING|ERROR)"
            echo "  --info                 Show hardware info and exit"
            echo "  --validate             Validate configuration and exit"
            echo "  --help                Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Build Python command
PYTHON_CMD="python3 $PROJECT_DIR/scripts/launcher/launch.py"

# Add arguments
PYTHON_CMD+=" --model $MODEL"
PYTHON_CMD+=" --preset $PRESET"

if [ "$ENABLE_API" = false ]; then
    PYTHON_CMD+=" --no-api"
fi

if [ "$ENABLE_UI" = false ]; then
    PYTHON_CMD+=" --no-ui"
fi

PYTHON_CMD+=" --api-host $API_HOST"
PYTHON_CMD+=" --api-port $API_PORT"
PYTHON_CMD+=" --ui-port $UI_PORT"
PYTHON_CMD+=" --log-level $LOG_LEVEL"

if [ -n "$CONFIG_FILE" ]; then
    PYTHON_CMD+=" --config $CONFIG_FILE"
fi

if [ -n "$SAVE_CONFIG" ]; then
    PYTHON_CMD+=" --save-config $SAVE_CONFIG"
fi

if [ "$INFO_MODE" = true ]; then
    PYTHON_CMD+=" --info"
fi

if [ "$VALIDATE_MODE" = true ]; then
    PYTHON_CMD+=" --validate"
fi

# Run the command
echo "Starting Apourvoir Optimized Launcher..."
echo "Command: $PYTHON_CMD"
echo ""

# Check if Python is available
if command -v python3 &> /dev/null; then
    exec $PYTHON_CMD
elif command -v python &> /dev/null; then
    exec python "$PROJECT_DIR/scripts/launcher/launch.py" $@
else
    echo "Error: Python is not installed"
    exit 1
fi
