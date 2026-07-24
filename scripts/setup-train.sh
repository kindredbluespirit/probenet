#!/usr/bin/env bash
# ProbeNet training environment setup for Lambda Labs / cloud GPU machines.
#
# Installs uv, syncs the root project with training extras, downloads assets.
#
# Usage:
#   bash scripts/setup-train.sh
#
# Environment variables:
#   PROBENET_ASSETS_ROOT  — override asset destination (default: <repo_root>/assets)
#   UV_LINK_MODE          — set to "copy" for Docker-style installs (default: copy)

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

echo "========================================"
echo " ProbeNet — Training environment setup"
echo "========================================"

# ── 1. Install uv if missing ──────────────────────────────────────────────────

if ! command -v uv &>/dev/null; then
    echo "[1/5] Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "[1/5] uv already installed ($(uv --version))"
fi

# ── 2. Install system packages ────────────────────────────────────────────────

echo "[2/5] Ensuring system dependencies..."
sudo apt-get update -qq && sudo apt-get install -y -qq \
    curl git git-lfs build-essential cmake \
    libgl1-mesa-dev libglfw3 libglfw3-dev \
    ffmpeg \
    > /dev/null && \
    sudo rm -rf /var/lib/apt/lists/*

# ── 3. Sync root project (training extras) ────────────────────────────────────

echo "[3/5] Syncing root project with training extras..."
UV_LINK_MODE="${UV_LINK_MODE:-copy}" uv sync --extra training --extra dev

# ── 4. Download assets ────────────────────────────────────────────────────────

echo "[4/5] Downloading SO-101 assets..."
bash scripts/download_assets.sh

# ── 5. Verify ─────────────────────────────────────────────────────────────────

echo "[5/5] Verifying installation..."
uv run python -c "
import probenet
import openpi
print(f'probenet: {probenet.__version__}')

import torch
print(f'torch:    {torch.__version__}')
print(f'CUDA:     {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device:   {torch.cuda.get_device_name(0)}')
"

echo ""
echo "========================================"
echo " Setup complete!"
echo ""
echo " Next steps:"
echo "   python scripts/train.py --config-name pi05_so101 --exp-name baseline_v1"
echo "========================================"
