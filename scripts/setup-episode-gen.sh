#!/usr/bin/env bash
# ProbeNet Isaac Sim data-generation environment setup for Lambda Labs.
#
# Installs uv, system deps, Isaac Sim pip packages, SO-101 assets.
# Designed for headless GPU cloud instances (no display required).
#
# Usage:
#   bash scripts/setup-episode-gen.sh
#
# Environment variables:
#   PROBENET_ASSETS_ROOT  — override asset destination (default: <repo_root>/assets)
#   UV_LINK_MODE          — set to "copy" for Docker-style installs (default: copy)
#   OMNI_KIT_ACCEPT_EULA  — must be "YES" to install Isaac Sim (set automatically)

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

echo "========================================"
echo " ProbeNet — Isaac Sim data gen setup"
echo "========================================"

# ── 1. Install uv if missing ──────────────────────────────────────────────────

if ! command -v uv &>/dev/null; then
    echo "[1/6] Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "[1/6] uv already installed ($(uv --version))"
fi

# ── 2. Install system packages ────────────────────────────────────────────────

echo "[2/6] Installing system dependencies (including Isaac Sim prerequisites)..."
sudo apt-get update -qq && sudo apt-get install -y -qq \
    curl git git-lfs build-essential cmake \
    libgl1-mesa-dev libglfw3 libglfw3-dev \
    libxi-dev libglew-dev xorg-dev \
    ffmpeg libxinerama-dev libxcursor1 libxrandr2 \
    > /dev/null && \
    sudo rm -rf /var/lib/apt/lists/*

# ── 3. Sync root project ──────────────────────────────────────────────────────

echo "[3/6] Syncing root project..."
UV_LINK_MODE="${UV_LINK_MODE:-copy}" uv sync --frozen --no-dev

# ── 4. Sync episode_gen/sim environment ───────────────────────────────────────

echo "[4/6] Syncing Isaac Sim environment..."
cd episode_gen/sim
UV_LINK_MODE="${UV_LINK_MODE:-copy}" uv sync --frozen
cd "$REPO_ROOT"

# ── 5. Install Isaac Sim (with NVIDIA PyPI index) ─────────────────────────────

echo "[5/6] Installing Isaac Sim..."
export OMNI_KIT_ACCEPT_EULA=YES

uv pip install --upgrade torch==2.11.0 \
    --index-url https://download.pytorch.org/whl/cu128

uv pip install "isaacsim[all,extscache]==6.0.1.0" \
    --extra-index-url https://pypi.nvidia.com \
    --index-strategy unsafe-best-match \
    --prerelease=allow

# ── 6. Download assets + verify ───────────────────────────────────────────────

echo "[6/6] Downloading SO-101 assets and verifying..."
bash scripts/download_assets.sh

uv run python -c "
import probenet
print(f'probenet: {probenet.__version__}')

import torch
print(f'torch:    {torch.__version__}')
print(f'CUDA:     {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device:   {torch.cuda.get_device_name(0)}')

print('Checking Isaac Sim env registration...')
from isaacsim import SimulationApp
print('SimulationApp imported OK')
"

echo ""
echo "========================================"
echo " Setup complete!"
echo ""
echo " Next steps:"
echo "   cd episode_gen/sim"
echo "   uv run python server.py \\"
echo "     --headless --record ../../data/lerobot/so101_pick_place \\"
echo "     --num-episodes 50"
echo "========================================"
