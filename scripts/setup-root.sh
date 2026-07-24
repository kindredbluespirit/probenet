#!/usr/bin/env bash
# ProbeNet root orchestrator venv setup.
# Installs uv if missing, then syncs the root project (probenet core).
#
# Usage:
#   bash scripts/setup-root.sh
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

echo "========================================"
echo " ProbeNet — root orchestrator venv"
echo "========================================"

if ! command -v uv &>/dev/null; then
    echo "[1/2] Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "[1/2] uv already installed ($(uv --version))"
fi

echo "[2/2] Syncing root project (no dev)..."
UV_LINK_MODE="${UV_LINK_MODE:-copy}" uv sync --frozen --no-dev

echo ""
echo " Verifying..."
uv run python -c "
import probenet
print(f'probenet: {probenet.__version__}')
"
echo " Root venv ready!"
