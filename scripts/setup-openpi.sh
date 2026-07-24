#!/usr/bin/env bash
# ProbeNet openpi (π₀.₅) inference server venv setup.
# Installs uv if missing, then syncs the policies/openpi/ submodule.
#
# Usage:
#   bash scripts/setup-openpi.sh
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"

echo "========================================"
echo " ProbeNet — openpi inference venv"
echo "========================================"

if ! command -v uv &>/dev/null; then
    echo "[1/2] Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "[1/2] uv already installed ($(uv --version))"
fi

echo "[2/2] Syncing policies/openpi/..."
cd "$REPO_ROOT/policies/openpi"
UV_LINK_MODE="${UV_LINK_MODE:-copy}" uv sync --no-dev

echo ""
echo " Verifying..."
uv run python -c "
from openpi.policies import policy_config
from openpi.serving import websocket_policy_server
print('openpi imports OK')
print('Use: uv run python scripts/serve_policy.py --help')
"
echo " openpi venv ready!"
