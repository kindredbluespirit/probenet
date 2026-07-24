#!/usr/bin/env bash
# ProbeNet real SO-101 robot data generation environment setup.
#
# Installs uv, syncs root project + episode_gen/so101, downloads assets.
# Lightweight — no Isaac Sim dependencies.
#
# Usage:
#   bash scripts/setup-episode-gen-so101.sh
#
# Environment variables:
#   PROBENET_ASSETS_ROOT  — override asset destination (default: <repo_root>/assets)
#   UV_LINK_MODE          — set to "copy" for Docker-style installs (default: copy)

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

echo "========================================"
echo " ProbeNet — SO-101 data gen setup"
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

# ── 3. Sync root project + SO-101 backend ─────────────────────────────────────

echo "[3/5] Syncing root project..."
UV_LINK_MODE="${UV_LINK_MODE:-copy}" uv sync

echo "       Syncing episode_gen/so101..."
cd episode_gen/so101
UV_LINK_MODE="${UV_LINK_MODE:-copy}" uv sync
cd "$REPO_ROOT"

# ── 4. Install hf CLI ─────────────────────────────────────────────────────────

echo "[4/5] Installing Hugging Face CLI..."
if ! command -v hf &>/dev/null; then
    curl -LsSf https://hf.co/cli/install.sh | bash
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "       hf CLI already installed ($(hf --version | head -1))"
fi

# ── 5. Download assets + verify ───────────────────────────────────────────────

echo "[5/5] Downloading SO-101 assets and verifying..."
bash scripts/download_assets.sh

uv run python -c "
import probenet
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
echo "   cd episode_gen/so101"
echo "   uv run python server.py --port 8227"
echo "========================================"
