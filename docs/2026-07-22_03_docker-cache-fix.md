# Docker Build Cache — 2026-07-22

## Problem

The `docker-publish` workflow took ~1h40m per run because every build started
from scratch. Three issues prevented caching:

1. **No `cache-to`** — only `cache-from: type=gha` was set, but nothing ever
   wrote back to the cache. The previous `cache-to` was removed due to "default
   Docker driver doesn't support cache export" — which happens when
   `setup-buildx-action` is missing and the legacy `docker` driver is used
   instead of BuildKit's `docker-container` driver.

2. **`COPY . .` before `uv sync`** — any source file change invalidated the
   layer containing the 30 GB `uv sync` + `isaacsim` install, forcing a full
   re-download on every push.

3. **No cache scope separation** — the trainer and rollout matrix jobs would
   share the same cache key, causing potential conflicts.

## Solution

### 1. Buildx setup + cache-to (`docker-publish.yml`)

```yaml
- name: Set up Docker Buildx
  uses: docker/setup-buildx-action@v3

- name: Build and push
  uses: docker/build-push-action@v6
  with:
    cache-from: type=gha,scope=${{ matrix.target }}
    cache-to: type=gha,mode=max,scope=${{ matrix.target }}
```

`setup-buildx-action@v3` provisions BuildKit with the `docker-container`
driver, which supports cache export. `scope` isolates trainer vs rollout
cache entries.

### 2. Layered COPY in Dockerfile

A new `deps` intermediate stage copies only the files needed for dependency
resolution (pyproject.toml, uv.lock, .python-version, README.md, backends/).
This layer is cached unless those files change.

Trainer and rollout inherit from `deps`. Source code (`COPY . .`) is copied
after the expensive installs, so only the final fast step re-runs on source
changes:

**Trainer:**
```dockerfile
FROM deps AS trainer
COPY . .
RUN uv sync --frozen --no-dev     # updates probenet only (~0.7ms)
```

**Rollout:**
```dockerfile
FROM deps AS rollout
RUN apt-get install ... xorg-dev ...
RUN uv pip install ... torch==2.11.0 + isaacsim  # cached
COPY . .
RUN uv pip install --no-deps -e .                # probenet only (~0.6ms)
```

Rollout uses `uv pip install --no-deps -e .` instead of `uv sync --frozen`
because sync would downgrade torch from 2.11.0 back to the lockfile's 2.7.1.

## Cache Invalidation Rules

| What changes | Layers rebuilt |
|---|---|
| Source files (`src/`, `scripts/`, etc.) | Final `uv pip install --no-deps -e .` only (~0.6ms) |
| `pyproject.toml`, `uv.lock` | Full deps stage + everything below |
| `backends/openpi` (submodule) | Full deps stage + everything below |

## Expected Build Times

| Scenario | Time |
|---|---|
| First build (cold cache) | ~1h40m (unchanged) |
| Source-only change (warm cache) | ~2-3 min |
| Dep change (warm cache) | ~1h40m (unchanged) |
