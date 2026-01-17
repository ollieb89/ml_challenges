#!/bin/bash
# Development session manager for parallel ML pipeline development
# Creates a tmux session with separate windows for pose analyzer, GPU optimizer, and monitoring

set -e

SESSION_NAME="ml"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    echo "Error: tmux is not installed. Please install it first:"
    echo "  Ubuntu/Debian: sudo apt-get install tmux"
    echo "  Fedora/RHEL: sudo dnf install tmux"
    echo "  macOS: brew install tmux"
    exit 1
fi

# Check if session already exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Session '$SESSION_NAME' already exists."
    read -p "Do you want to (a)ttach, (k)ill and recreate, or (c)ancel? [a/k/c]: " choice
    case "$choice" in
        a|A)
            tmux attach -t "$SESSION_NAME"
            exit 0
            ;;
        k|K)
            echo "Killing existing session..."
            tmux kill-session -t "$SESSION_NAME"
            ;;
        *)
            echo "Cancelled."
            exit 0
            ;;
    esac
fi

echo "Creating tmux session '$SESSION_NAME' for ML pipeline development..."

# Create new detached session
tmux new-session -d -s "$SESSION_NAME" -x 180 -y 50 -c "$PROJECT_ROOT"

# Window 0: Pose Analyzer Development
tmux rename-window -t "$SESSION_NAME:0" "pose"
tmux send-keys -t "$SESSION_NAME:pose" "cd $PROJECT_ROOT/projects/pose_analyzer" Enter
tmux send-keys -t "$SESSION_NAME:pose" "echo '=== Pose Analyzer Development Environment ==='" Enter
tmux send-keys -t "$SESSION_NAME:pose" "echo 'Starting IPython with pixi environment...'" Enter
tmux send-keys -t "$SESSION_NAME:pose" "pixi run python -m IPython" Enter

# Window 1: GPU Optimizer Development
tmux new-window -t "$SESSION_NAME" -n "vram" -c "$PROJECT_ROOT/projects/gpu_optimizer"
tmux send-keys -t "$SESSION_NAME:vram" "echo '=== GPU Optimizer Development Environment ==='" Enter
tmux send-keys -t "$SESSION_NAME:vram" "echo 'Starting IPython with pixi environment...'" Enter
tmux send-keys -t "$SESSION_NAME:vram" "pixi run python -m IPython" Enter

# Window 2: Monitoring & Testing
tmux new-window -t "$SESSION_NAME" -n "monitor" -c "$PROJECT_ROOT"
tmux send-keys -t "$SESSION_NAME:monitor" "echo '=== System Monitoring & Testing ==='" Enter
tmux send-keys -t "$SESSION_NAME:monitor" "echo 'GPU Status:'" Enter
tmux send-keys -t "$SESSION_NAME:monitor" "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv" Enter
tmux send-keys -t "$SESSION_NAME:monitor" "echo ''" Enter
tmux send-keys -t "$SESSION_NAME:monitor" "echo 'Starting continuous monitoring (Ctrl+C to stop)...'" Enter
tmux send-keys -t "$SESSION_NAME:monitor" "sleep 2" Enter
tmux send-keys -t "$SESSION_NAME:monitor" "watch -n 1 'nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits'" Enter

# Window 3: General terminal
tmux new-window -t "$SESSION_NAME" -n "terminal" -c "$PROJECT_ROOT"
tmux send-keys -t "$SESSION_NAME:terminal" "echo '=== General Terminal ==='" Enter
tmux send-keys -t "$SESSION_NAME:terminal" "echo 'Project root: $PROJECT_ROOT'" Enter
tmux send-keys -t "$SESSION_NAME:terminal" "echo ''" Enter
tmux send-keys -t "$SESSION_NAME:terminal" "echo 'Quick commands:'" Enter
tmux send-keys -t "$SESSION_NAME:terminal" "echo '  pixi run pytest                    # Run all tests'" Enter
tmux send-keys -t "$SESSION_NAME:terminal" "echo '  pixi run pytest projects/pose_analyzer/tests  # Test pose analyzer'" Enter
tmux send-keys -t "$SESSION_NAME:terminal" "echo '  pixi run pytest projects/gpu_optimizer/tests  # Test GPU optimizer'" Enter
tmux send-keys -t "$SESSION_NAME:terminal" "echo '  ./scripts/validate_env.py          # Validate environment'" Enter
tmux send-keys -t "$SESSION_NAME:terminal" "echo ''" Enter

# Select the first window
tmux select-window -t "$SESSION_NAME:0"

# Display usage information
cat << 'EOF'

╔════════════════════════════════════════════════════════════════╗
║         ML Pipeline Development Session Created                ║
╚════════════════════════════════════════════════════════════════╝

Session: ml

Windows:
  0: pose     - Pose Analyzer IPython environment
  1: vram     - GPU Optimizer IPython environment  
  2: monitor  - nvidia-smi monitoring (watch mode)
  3: terminal - General terminal for testing/validation

Tmux Quick Reference:
  Ctrl+b c        Create new window
  Ctrl+b n        Next window
  Ctrl+b p        Previous window
  Ctrl+b 0-9      Switch to window number
  Ctrl+b d        Detach from session
  Ctrl+b [        Enter scroll mode (q to exit)
  Ctrl+b ?        Show all keybindings

Reattach: tmux attach -t ml
Kill session: tmux kill-session -t ml

Attaching to session...

EOF

# Attach to the session
tmux attach -t "$SESSION_NAME"
