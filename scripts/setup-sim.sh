#!/usr/bin/env bash
# ProbeNet Isaac Sim data generation venv setup.
# Installs uv, system deps, and syncs episode_gen/sim/.
#
# Usage:
#   bash scripts/setup-sim.sh
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"

echo "========================================"
echo " ProbeNet — Isaac Sim data gen venv"
echo "========================================"

if ! command -v uv &>/dev/null; then
    echo "[1/3] Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "[1/3] uv already installed ($(uv --version))"
fi

echo "[2/3] Installing system dependencies..."
sudo apt-get update -qq && sudo apt-get install -y -qq \
    curl git git-lfs build-essential cmake \
    libgl1-mesa-dev libglfw3 libglfw3-dev \
    libxi-dev libglew-dev xorg-dev \
    ffmpeg libxinerama-dev libxcursor1 libxrandr2 \
    > /dev/null && \
    sudo rm -rf /var/lib/apt/lists/*

echo "[3/3] Syncing episode_gen/sim/..."
cd "$REPO_ROOT/episode_gen/sim"
UV_LINK_MODE="${UV_LINK_MODE:-copy}" uv sync --no-dev

echo ""
echo " Verifying..."
uv run python -c "
import probenet
import torch
print(f'probenet: {probenet.__version__}')
print(f'torch:    {torch.__version__}')
# Isaac Sim import is deferred to runtime (headless GPU)
print('Use: uv run python server.py --help')
"
echo " Sim venv ready!"
