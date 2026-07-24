#!/usr/bin/env bash
# ProbeNet GR00T N1.7 inference server venv setup.
# Installs uv if missing, then syncs the policies/gr00t/ submodule.
#
# Usage:
#   bash scripts/setup-gr00t.sh
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"

echo "========================================"
echo " ProbeNet — GR00T inference venv"
echo "========================================"

if ! command -v uv &>/dev/null; then
    echo "[1/2] Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "[1/2] uv already installed ($(uv --version))"
fi

echo "[2/2] Syncing policies/gr00t/..."
cd "$REPO_ROOT/policies/gr00t"
UV_LINK_MODE="${UV_LINK_MODE:-copy}" uv sync --no-dev

echo ""
echo " Verifying..."
uv run python -c "
# Phase E — placeholder until GR00T server is fully integrated.
print('GR00T venv synced.')
print('Use: uv run python gr00t/eval/run_gr00t_server.py --help')
"
echo " GR00T venv ready!"
