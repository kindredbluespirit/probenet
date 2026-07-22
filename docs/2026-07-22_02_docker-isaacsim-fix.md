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

## Verification

Confirmed via dry-run locally:

```bash
uv pip install --dry-run "isaacsim[all,extscache]==6.0.1.0" \
    --extra-index-url https://pypi.nvidia.com \
    --index-strategy unsafe-best-match \
    --prerelease=allow
```
