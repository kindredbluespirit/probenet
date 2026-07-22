# Docker Rollout Stage Fix

**Date:** 2026-07-22  
**Issue:** GitHub Actions `docker-publish` workflow failing on rollout target build.

## Root Cause

The `isaacsim[all,extscache]==6.0.1.0` pip install was failing during dependency resolution with two distinct errors:

1. **`mujoco-usd-converter==0.2.0` not found** — This package exists on PyPI but is first discovered via the NVIDIA extra index (`pypi.nvidia.com`), which only has a different version. uv's default index strategy (`first-match`) stops at the first index that hosts any version of a package, never checking subsequent indexes.

2. **`tinyobjloader==2.0.0rc13` not found** — This is a pre-release version. uv excludes pre-releases by default.

Both are transitive dependencies of `isaacsim-core==6.0.1.0`.

## Fix

Added two flags to the isaacsim `uv pip install` command in `docker/Dockerfile`:

```dockerfile
uv pip install "isaacsim[all,extscache]==6.0.1.0" \
    --extra-index-url https://pypi.nvidia.com \
    --index-strategy unsafe-best-match \
    --prerelease=allow
```

- `--index-strategy unsafe-best-match` — Searches all configured indexes for the best version of each package, not just the first matching index.
- `--prerelease=allow` — Enables installation of pre-release versions.

## Second Failure: `import isaacsim` at Build Time

After fixing the install flags, the build failed at `RUN uv run python -c "import isaacsim; import probenet"`:

- `import isaacsim` triggers Omniverse Kit GPU initialization — CI runners have no GPU, so this always fails.
- `ENV OMNI_KIT_ACCEPT_EULA=YES` was set **after** the import (line 59 vs line 57), so EULA wasn't accepted.

The probenet package itself never imports isaacsim — it's only used at runtime with a GPU.

### Fix

- Moved `ENV OMNI_KIT_ACCEPT_EULA=YES` **before** the import check.
- Changed smoke test to `import probenet` only (isaacsim can't import without GPU).

```dockerfile
ENV OMNI_KIT_ACCEPT_EULA=YES
RUN uv run python -c "import probenet"
```

## Third Failure: CI `ModuleNotFoundError: No module named 'probenet.env'`

The CI workflow (`ci.yml`) had a redundant `uv pip install -e .` step after `uv sync --extra dev`. Running `uv pip install` on a `uv sync`-managed venv can break package installations — the same class of conflict as the Docker `uv pip install` after `uv sync` pattern.

### Fix

Removed the redundant `uv pip install -e .` — `uv sync --extra dev` already installs probenet:

```diff
  - name: Install dependencies
    run: uv sync --extra dev
- - name: Install package
-   run: uv pip install -e .
  - name: Run tests
    run: uv run pytest -v
```

## Verification

isaacsim install flags confirmed via dry-run locally:

```bash
uv pip install --dry-run "isaacsim[all,extscache]==6.0.1.0" \
    --extra-index-url https://pypi.nvidia.com \
    --index-strategy unsafe-best-match \
    --prerelease=allow
```
