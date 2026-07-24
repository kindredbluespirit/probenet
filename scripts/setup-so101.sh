#!/usr/bin/env bash
# ProbeNet real SO-101 robot data generation venv setup.
# Installs uv if missing, then syncs episode_gen/so101/.
#
# Usage:
#   bash scripts/setup-so101.sh
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"

echo "========================================"
echo " ProbeNet — SO-101 robot data gen venv"
echo "========================================"

if ! command -v uv &>/dev/null; then
    echo "[1/2] Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "[1/2] uv already installed ($(uv --version))"
fi

echo "[2/2] Syncing episode_gen/so101/..."
cd "$REPO_ROOT/episode_gen/so101"
UV_LINK_MODE="${UV_LINK_MODE:-copy}" uv sync --no-dev

echo ""
echo " Verifying..."
uv run python -c "
import probenet
print(f'probenet: {probenet.__version__}')
print('Use: uv run python server.py --help')
"
echo " SO-101 venv ready!"
