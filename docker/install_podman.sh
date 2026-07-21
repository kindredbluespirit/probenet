#!/usr/bin/env bash
# ProbeNet Lambda instance bootstrap — podman + NVIDIA CDI
#
# Run once on a fresh Lambda Labs Ubuntu 24.04 instance before pulling the
# probenet Docker image.
#
# Usage:
#   curl -sSL https://raw.githubusercontent.com/.../install_podman.sh | bash

set -euo pipefail

GREEN="\033[0;32m"
NC="\033[0m"

step() { echo -e "${GREEN}[$1]${NC} $2"; }

step 1 "Install podman + NVIDIA container toolkit"
sudo apt-get update -qq
sudo apt-get install -y -qq podman nvidia-container-toolkit > /dev/null
podman --version && echo "  podman OK"

step 2 "Generate NVIDIA CDI config"
sudo nvidia-ctk cdi generate --output /etc/cdi/nvidia.yaml > /dev/null 2>&1
podman run --rm --device nvidia.com/gpu=all \
    nvcr.io/nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi -L > /dev/null && \
    echo "  GPU access OK" || echo "  WARNING: GPU test failed"

step 3 "Login to GitHub Container Registry"
if [ -n "${GITHUB_TOKEN:-}" ]; then
    echo "$GITHUB_TOKEN" | podman login ghcr.io -u PROBENET --password-stdin
elif [ -n "${GH_TOKEN:-}" ]; then
    echo "$GH_TOKEN" | podman login ghcr.io -u PROBENET --password-stdin
else
    echo "  Skipped (set GITHUB_TOKEN or GH_TOKEN to auto-login)"
fi

step 4 "Create data directories"
mkdir -p ~/probenet-data ~/probenet-outputs ~/isaac-sim-assets
echo "  ~/probenet-data, ~/probenet-outputs, ~/isaac-sim-assets"

echo ""
echo "================================================================"
echo "  ProbeNet Lambda instance ready."
echo ""
echo "  Pull image:  podman pull ghcr.io/<user>/probenet:trainer"
echo "  Pull image:  podman pull ghcr.io/<user>/probenet:rollout"
echo ""
echo "  Run trainer:"
echo "    podman run --device nvidia.com/gpu=all \\"
echo "      -v ~/probenet-data:/data \\"
echo "      -v ~/probenet-outputs:/workspace/probenet/outputs \\"
echo "      -e HF_TOKEN=\$HF_TOKEN \\"
echo "      ghcr.io/<user>/probenet:trainer"
echo ""
echo "  Run rollout:"
echo "    podman run --device nvidia.com/gpu=all \\"
echo "      -v ~/probenet-data:/data \\"
echo "      -v ~/probenet-outputs:/workspace/probenet/outputs \\"
echo "      -v ~/isaac-sim-assets:/isaac-sim/assets:ro \\"
echo "      -e HF_TOKEN=\$HF_TOKEN \\"
echo "      ghcr.io/<user>/probenet:rollout"
echo "================================================================"
