# ProbeNet — multi-target Dockerfile
#
# Build targets:
#   podman build --target trainer -t probenet:trainer .
#   podman build --target rollout -t probenet:rollout .
#
# Trainer: openpi + probenet, torch 2.7.1, JAX, flax (~10 GB)
# Rollout: probenet + Isaac Sim pip, torch 2.11.0 (~30 GB)

# ── Base ────────────────────────────────────────────────────────────────────

FROM nvidia/cuda:12.6.0-cudnn-devel-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

RUN apt-get update -qq && apt-get install -y -qq \
    python3.12 python3.12-dev python3.12-venv \
    curl git git-lfs build-essential cmake \
    libgl1-mesa-dev libglfw3 libglfw3-dev \
    ffmpeg libxinerama-dev libxcursor1 libxrandr2 \
    > /dev/null && \
    rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
    python -m ensurepip --upgrade && \
    curl -LsSf https://astral.sh/uv/install.sh | sh

ENV PATH="/root/.local/bin:$PATH"
WORKDIR /workspace/probenet

# ── Trainer ─────────────────────────────────────────────────────────────────

FROM base AS trainer

COPY backends/openpi/ /workspace/probenet/backends/openpi/
COPY pyproject.toml uv.lock ./
COPY src/ ./src/

RUN uv sync --frozen --no-dev
RUN uv run python -c "import openpi; import probenet"

ENTRYPOINT ["uv", "run", "python", "-m", "probenet.cli", "--mode", "trainer"]

# ── Rollout ─────────────────────────────────────────────────────────────────

FROM base AS rollout

RUN apt-get update -qq && apt-get install -y -qq \
    libxi-dev libglew-dev xorg-dev \
    > /dev/null && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
COPY src/ ./src/

RUN uv sync --frozen --no-dev && \
    uv pip install --upgrade torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128 && \
    uv pip install "isaacsim[all,extscache]==6.0.1.0" --extra-index-url https://pypi.nvidia.com

RUN uv run python -c "import isaacsim; import probenet"

ENV OMNI_KIT_ACCEPT_EULA=YES
ENTRYPOINT ["uv", "run", "python", "-m", "probenet.cli", "--mode", "rollout"]
