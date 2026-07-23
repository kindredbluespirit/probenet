# ProbeNet — multi-target Dockerfile
#
# Build targets:
#   podman build --target trainer -t probenet:trainer .
#   podman build --target rollout -t probenet:rollout .
#
# Trainer: openpi + probenet, torch 2.7.1, JAX, flax (~10 GB)
# Rollout: probenet + Isaac Sim 6.0.1, torch 2.11.0 (~30 GB)
#
# Rollout uses two virtual environments:
#   .venv          — openpi policy server (torch 2.7.1 from lockfile)
#   .venv-isaac    — Isaac Sim rollout worker (torch 2.11.0)
# The policy server and Isaac Sim worker run as separate processes
# communicating over WebSocket, so there's no torch version conflict.
#
# Caching: dependency files are copied before source so the expensive
# uv sync layers are only invalidated when deps change.
# Source-only changes rebuild only the final fast layer.

# ── Base ────────────────────────────────────────────────────────────────────

FROM nvidia/cuda:12.6.0-cudnn-devel-ubuntu24.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_PYTHON_PREFERENCE=only-managed

RUN apt-get update -qq && apt-get install -y -qq \
    curl git git-lfs build-essential cmake \
    libgl1-mesa-dev libglfw3 libglfw3-dev \
    ffmpeg libxinerama-dev libxcursor1 libxrandr2 && \
    rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

ENV PATH="/root/.local/bin:$PATH"
WORKDIR /workspace/probenet

# ── Deps (cached unless pyproject/uv.lock/backends change) ───────────────────

FROM base AS deps

COPY pyproject.toml uv.lock .python-version README.md ./
COPY backends/ backends/

RUN uv sync --frozen --no-dev

# ── Trainer ─────────────────────────────────────────────────────────────────

FROM deps AS trainer

COPY . .
RUN uv run python -c "import openpi; import probenet"

ENTRYPOINT ["uv", "run", "python", "-m", "probenet.cli", "--mode", "trainer"]

# ── Rollout ─────────────────────────────────────────────────────────────────

FROM deps AS rollout

RUN apt-get update -qq && apt-get install -y -qq \
    libxi-dev libglew-dev xorg-dev \
    > /dev/null && \
    rm -rf /var/lib/apt/lists/*

# Isaac Sim isolation venv (torch 2.11.0 required by 6.0.1)
RUN uv venv .venv-isaac && \
    uv pip install --python .venv-isaac --upgrade torch==2.11.0 \
    --index-url https://download.pytorch.org/whl/cu128 && \
    uv pip install --python .venv-isaac \
    "isaacsim[all,extscache]==6.0.1.0" \
    --extra-index-url https://pypi.nvidia.com \
    --index-strategy unsafe-best-match --prerelease=allow

COPY . .
RUN uv pip install --no-deps -e . && \
    uv pip install --python .venv-isaac --no-deps -e .

ENV OMNI_KIT_ACCEPT_EULA=YES
RUN .venv-isaac/bin/python -c "import probenet"

ENTRYPOINT ["uv", "run", "python", "-m", "probenet.cli", "--mode", "rollout"]
