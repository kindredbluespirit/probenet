#!/usr/bin/env bash
# ProbeNet Isaac Sim data generation environment setup.
#
# Installs uv, system deps, Isaac Sim (pip from NVIDIA index), SO-101 assets.
# Designed for headless GPU cloud instances.
#
# Usage:
#   bash scripts/setup-episode-gen-sim.sh
#
# Environment variables:
#   PROBENET_ASSETS_ROOT  — override asset destination (default: <repo_root>/assets)
#   UV_LINK_MODE          — set to "copy" for Docker-style installs (default: copy)

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

echo "========================================"
echo " ProbeNet — Isaac Sim data gen setup"
echo "========================================"

# ── 1. Install uv if missing ──────────────────────────────────────────────────

if ! command -v uv &>/dev/null; then
    echo "[1/7] Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "[1/7] uv already installed ($(uv --version))"
fi

# ── 2. Install system packages (Isaac Sim needs extra GL/Xorg libs) ──────────

echo "[2/7] Installing system dependencies..."
sudo apt-get update -qq && sudo apt-get install -y -qq \
    curl git git-lfs build-essential cmake \
    libgl1-mesa-dev libglfw3 libglfw3-dev \
    libxi-dev libglew-dev xorg-dev \
    ffmpeg libxinerama-dev libxcursor1 libxrandr2 \
    > /dev/null && \
    sudo rm -rf /var/lib/apt/lists/*

# ── 3. Sync root project (no dev deps) ────────────────────────────────────────

echo "[3/7] Syncing root project..."
UV_LINK_MODE="${UV_LINK_MODE:-copy}" uv sync --frozen --no-dev

# ── 4. Install hf CLI ─────────────────────────────────────────────────────────

echo "[4/7] Installing Hugging Face CLI..."
if ! command -v hf &>/dev/null; then
    curl -LsSf https://hf.co/cli/install.sh | bash
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "       hf CLI already installed ($(hf --version | head -1))"
fi

# ── 5. Install Isaac Sim + dependencies ───────────────────────────────────────

echo "[5/7] Installing Isaac Sim (torch 2.11.0 cu128)..."
export OMNI_KIT_ACCEPT_EULA=YES

uv pip install --upgrade torch==2.11.0 \
    --index-url https://download.pytorch.org/whl/cu128

echo "       Installing isaacsim + isaaclab from NVIDIA index..."
uv pip install "isaacsim[all,extscache]==6.0.1.0" \
    --extra-index-url https://pypi.nvidia.com \
    --index-strategy unsafe-best-match \
    --prerelease=allow

uv pip install "isaaclab[isaacsim,all]>=2.3.0" \
    --extra-index-url https://pypi.nvidia.com \
    --index-strategy unsafe-best-match

# ── 6. Download assets ────────────────────────────────────────────────────────

echo "[6/7] Downloading SO-101 assets..."
bash scripts/download_assets.sh

# ── 7. Verify ─────────────────────────────────────────────────────────────────

echo "[7/7] Verifying installation..."
uv run python -c "
import probenet
print(f'probenet: {probenet.__version__}')

import torch
print(f'torch:    {torch.__version__}')
print(f'CUDA:     {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device:   {torch.cuda.get_device_name(0)}')

print('Isaac Sim import check...')
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
