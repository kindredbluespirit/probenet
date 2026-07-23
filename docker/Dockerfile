# ProbeNet — multi-target Dockerfile
#
# Build targets:
#   podman build --target trainer -t probenet:trainer .
#   podman build --target rollout -t probenet:rollout .
#
# Trainer: openpi + probenet, torch 2.7.1, JAX, flax (~10 GB)
# Rollout: probenet + Isaac Sim pip 5.1.0, torch 2.7.1 (~30 GB)
#
# Caching: dependency files are copied before source so the expensive
# uv sync + pip install layer is only invalidated when deps change.
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

RUN uv pip install "isaacsim[all,extscache]==5.1.0.0" \
    --extra-index-url https://pypi.nvidia.com \
    --index-strategy unsafe-best-match --prerelease=allow

COPY . .
RUN uv sync --frozen --no-dev

ENV OMNI_KIT_ACCEPT_EULA=YES
RUN uv run python -c "import probenet"
ENTRYPOINT ["uv", "run", "python", "-m", "probenet.cli", "--mode", "rollout"]
