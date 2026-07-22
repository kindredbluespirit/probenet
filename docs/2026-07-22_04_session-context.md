# Session Context — 2026-07-22

All fixes and changes made during this session, for continuing in a new conversation.

## Repo Layout
- Python: uv + hatchling, src layout (`src/probenet/`)
- Docker: multi-target Dockerfile (`docker/Dockerfile`), base `nvidia/cuda:12.6.0-cudnn-devel-ubuntu24.04`
- CI: `.github/workflows/ci.yml`, `.github/workflows/docker-publish.yml`, `.github/workflows/deploy.yml`
- Docs: `docs/YYYY-MM-DD_NN_n-a-m-e.md`

## Problems Fixed

### 1. Docker isaacsim install — missing deps on wrong index
**File:** `docker/Dockerfile:65`
**Error:** `mujoco-usd-converter==0.2.0` on wrong index, `tinyobjloader==2.0.0rc13` is pre-release
**Fix:** Added `--index-strategy unsafe-best-match --prerelease=allow` to isaacsim `uv pip install`

### 2. Docker import isaacsim at build time
**File:** `docker/Dockerfile:70-71`
**Error:** `import isaacsim` needs GPU, CI runner has none
**Fix:** Moved `ENV OMNI_KIT_ACCEPT_EULA=YES` before import, changed smoke test to `import probenet` only

### 3. CI pytest — ModuleNotFoundError: No module named 'probenet.env'
**Files:** `.github/workflows/ci.yml`, `.gitignore`
**Root cause:** `.gitignore` had `env/` which matched `src/probenet/env/` — the entire `env` package was never committed to git. CI checkout had no `env/` directory.
**Fix:** Changed `.gitignore` `env/` → `/env/` (root-only), then `git add` the env files. Also removed redundant `uv pip install -e .` from ci.yml (already covered by `uv sync --extra dev`).
**Files added to git:** `src/probenet/env/__init__.py`, `isaac_env.py`, `lerobot_adapter.py`, `real_robot.py`

### 4. Docker build time — 1h40m with no cache
**Files:** `docker/Dockerfile`, `.github/workflows/docker-publish.yml`
**Problems:**
- `cache-to` was removed because "default docker driver doesn't support cache export" — actually missing `setup-buildx-action`
- `COPY . .` before `uv sync` invalidated the entire 30GB install layer on every source change
- Trainer/rollout matrix jobs shared cache key (possible collisions)

**Fixes:**
1. Added `docker/setup-buildx-action@v3` to workflow
2. Added `cache-to: type=gha,mode=max,scope=${{ matrix.target }}` and scoped `cache-from`
3. Reorganized Dockerfile with `deps` intermediate stage:
   - Deps stage copies only `pyproject.toml`, `uv.lock`, `.python-version`, `README.md`, `backends/` → then `uv sync --frozen --no-dev`
   - Trainer inherits deps → `COPY . .` → `uv sync --frozen --no-dev` (fast re-sync)
   - Rollout inherits deps → xorg install → torch upgrade + isaacsim install → `COPY . .` → `uv pip install --no-deps -e .` (avoids sync downgrading torch)

**Why rollout uses `uv pip install --no-deps -e .` instead of `uv sync`:** `uv sync --frozen` syncs to lockfile which has `torch==2.7.1`. Running it after upgrading to torch 2.11.0 would downgrade torch back.

**Cache invalidation:** source-only changes → final layer only (~0.6ms). Dep file changes → full deps rebuild.

## Docs Created/Updated
- `docs/2026-07-22_02_docker-isaacsim-fix.md` — isaacsim install flags + import fix + CI fix
- `docs/2026-07-22_03_docker-cache-fix.md` — cache optimization details
- `docs/2026-07-22_01_cloud-docker-deployment.md` — updated cache section

## Current Run State
Workflow run 29937787066 ("Try reducing docker build time...") failed due to Ubuntu archive mirror outage (`Unable to connect to archive.ubuntu.com:80`) during apt-get in the base stage. Re-ran manually — in progress. First cold build ~1h40m.

## Unstaged Changes
- `.gitignore` — changed `env/` to `/env/`
- `src/probenet/env/` — staged for initial commit (4 files)

## Key Considerations for Continuing
- `src/probenet/env/` files need committing + pushing for CI to pass
- `uv pip install -e .` was removed from ci.yml — `uv sync --extra dev` covers it
- Rollout Docker stage uses `uv pip install --no-deps -e .` at end (not `uv sync`) to preserve torch 2.11.0
- GHA cache is 10GB per repo — rollout image may exceed this; if so, switch to `mode=min` or registry cache
- docker-publish triggers only on push to main (not PRs), so long builds are less disruptive
