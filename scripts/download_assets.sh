#!/usr/bin/env bash
# Downloads probenet simulation assets from the Hugging Face Storage Bucket.
#
# Usage:
#   bash scripts/download_assets.sh
#
# The assets are placed in assets/robots/so101_new_calib/
# Set PROBENET_ASSETS_ROOT to override the target directory.

set -euo pipefail

BUCKET="kindredbluespirit/so101-assets"
DEST="${PROBENET_ASSETS_ROOT:-$(git rev-parse --show-toplevel)/assets}"

echo "Downloading SO-101 robot USD from HF bucket '$BUCKET' to '$DEST/robots/'..."

mkdir -p "$DEST/robots"

if command -v hf &>/dev/null; then
    hf sync "hf://buckets/$BUCKET/robots/so101_new_calib" "$DEST/robots/so101_new_calib"
elif python3 -c "from huggingface_hub import snapshot_download" 2>/dev/null; then
    echo "Using huggingface_hub for download..."
    python3 -c "
from huggingface_hub import hf_hub_download
import os
dest = os.environ.get('PROBENET_ASSETS_ROOT', os.path.join(os.getcwd(), 'assets'))
# Download the main USD and its sublayers
files = [
    'so101_new_calib/so101_new_calib.usd',
    'so101_new_calib/configuration/so101_new_calib_base.usd',
    'so101_new_calib/configuration/so101_new_calib_physics.usd',
    'so101_new_calib/configuration/so101_new_calib_robot.usd',
    'so101_new_calib/configuration/so101_new_calib_sensor.usd',
]
for f in files:
    path = hf_hub_download(repo_id='$BUCKET', filename=f, local_dir=dest)
    print(f'Downloaded: {path}')
"
else
    echo "ERROR: Neither 'hf' CLI nor 'huggingface_hub' Python package found."
    echo "Install with: pip install huggingface_hub"
    exit 1
fi

echo "Done."
