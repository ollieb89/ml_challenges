#!/bin/bash
# Sync projects between machines via SSH using config/machines.yml

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${PROJECT_ROOT}/config/machines.yml"

# Optional first argument: machine key in machines.yml (desktop-5070ti, laptop-4070ti, pc-3070ti)
PROFILE="${1:-default}"

# Helper: get a value from machines.yml via python + PyYAML (no extra CLI deps)
get_yaml_value() {
  local profile="$1"
  local key="$2"

  python - <<'PYCODE' "$CONFIG_FILE" "$profile" "$key"
import sys, pathlib, yaml

config_path = pathlib.Path(sys.argv[1])
profile     = sys.argv[2]
key         = sys.argv[3]

with open(config_path) as f:
    cfg = yaml.safe_load(f)

machines = cfg.get("machines", {})
default_name = cfg.get("default")

name = profile
if name == "default":
    name = default_name

if not name or name not in machines:
    raise SystemExit(f"Could not resolve machine profile '{profile}' (resolved to '{name}') in {config_path}")

machine = machines[name]

value = machine
for part in key.split("."):
    if not isinstance(value, dict) or part not in value:
        value = None
        break
    value = value[part]

if value is None:
    raise SystemExit(f"Key '{key}' not found for machine '{name}'")

print(value)
PYCODE
}

# Resolve SSH info from config
SSH_USER="$(get_yaml_value "$PROFILE" 'ssh_user')"
SSH_HOST="$(get_yaml_value "$PROFILE" 'ssh_host')"
REMOTE_ROOT="$(get_yaml_value "$PROFILE" 'remote_root')"

REMOTE_HOST="${SSH_USER}@${SSH_HOST}"
REMOTE_BASE="${REMOTE_ROOT}"

echo "🔄 Syncing projects to profile '${PROFILE}' (${REMOTE_HOST})..."
echo "   Remote base path: ${REMOTE_BASE}"

# Sync pose_analyzer
rsync -avz --delete \
  "${PROJECT_ROOT}/projects/pose_analyzer/" \
  "${REMOTE_HOST}:${REMOTE_BASE}/projects/pose_analyzer/"

# Sync gpu_optimizer
rsync -avz --delete \
  "${PROJECT_ROOT}/projects/gpu_optimizer/" \
  "${REMOTE_HOST}:${REMOTE_BASE}/projects/gpu_optimizer/"

# Sync shared data
rsync -avz --delete \
  "${PROJECT_ROOT}/data/" \
  "${REMOTE_HOST}:${REMOTE_BASE}/data/"

echo "✅ Sync complete!"
