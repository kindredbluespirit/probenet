# Repository Restructure: Independent venvs + `backends/` → `policies/`

## Date
2026-07-24

## Motivation

The previous setup had two problems:

1. **Combined setup scripts** (`setup-episode-gen.sh`, `setup-episode-gen-sim.sh`,
   `setup-episode-gen-so101.sh`) tried to sync multiple venvs in one script. They were
   inconsistent — some synced a sub-env, others didn't — and the user had to run a single
   script for everything.

2. **`backends/` vs `policies/` confusion**: The actual model repos lived in `backends/`
   (as git submodules), while thin server wrappers lived in `policies/`. The wrappers
   duplicated what the submodules already provided (openpi has `scripts/serve_policy.py`
   built-in; GR00T has `gr00t/eval/run_gr00t_server.py`).

## What changed

### 1. `backends/` → `policies/` submodule rename

The two git submodules moved:
- `backends/openpi/` → `policies/openpi/`
- `backends/isaac-gr00t/` → `policies/gr00t/`

The thin wrapper directories `policies/openpi/serve.py`, `policies/openpi/pyproject.toml`,
`policies/gr00t/serve.py`, and `policies/gr00t/pyproject.toml` were removed. The
submodules' own `pyproject.toml` and their built-in serving scripts are used instead.

`.gitmodules`, `.git/config`, and `.git/modules/` were updated accordingly.

### 2. One setup script per venv

Five independent setup scripts replace the three old combined ones:

| Script | Target venv | Key deps |
|--------|------------|----------|
| `scripts/setup-root.sh` | `.venv` (repo root) | websockets, numpy, hf-hub, pyyaml, scipy |
| `scripts/setup-openpi.sh` | `policies/openpi/.venv` | jax, flax, orbax, openpi |
| `scripts/setup-gr00t.sh` | `policies/gr00t/.venv` | torch, gr00t |
| `scripts/setup-sim.sh` | `episode_gen/sim/.venv` | isaacsim, isaaclab, lerobot |
| `scripts/setup-so101.sh` | `episode_gen/so101/.venv` | lerobot, pyserial, pyrealsense2 |

Each script:
- Installs `uv` if missing
- Installs system deps when needed (sim needs GL/Xorg libs)
- Runs `uv sync --no-dev` in the target directory only
- Verifies with a quick import check

### 3. Deleted files

- `scripts/setup-episode-gen.sh`
- `scripts/setup-episode-gen-sim.sh`
- `scripts/setup-episode-gen-so101.sh`

## How to use

For a full deployment, run each needed venv's script independently:

```bash
# Clone + init submodules
git clone https://github.com/kindredbluespirit/probenet.git
cd probenet
git submodule update --init --recursive

# Set up the three venvs needed for Isaac Sim data generation:
bash scripts/setup-root.sh
bash scripts/setup-openpi.sh
bash scripts/setup-sim.sh

# Or for real robot data generation:
bash scripts/setup-root.sh
bash scripts/setup-openpi.sh
bash scripts/setup-so101.sh
```

## Files updated

| File | Change |
|------|--------|
| `.gitmodules` | `backends/openpi` → `policies/openpi`, `backends/isaac-gr00t` → `policies/gr00t` |
| `pyproject.toml` | openpi source path updated |
| `src/probenet/training/trainer.py` | GR00T launch path updated |
| `docs/2026-07-20_03_implementation-plan.md` | directory references |
| `docs/2026-07-21_01-status-report.md` | directory tree |
| `docs/2026-07-22_01_cloud-docker-deployment.md` | GR00T path |
| `docs/2026-07-23_01_multi-service-architecture.md` | directory references |
