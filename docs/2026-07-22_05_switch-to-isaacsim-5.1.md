# Switch to Isaac Sim 5.1.0 — Single Torch Version

## Background

The rollout Docker stage previously used **Isaac Sim 6.0.1.0**, which required
**torch 2.11.0**. This conflicted with the project's lockfile pin of
`torch==2.7.1` (required by openpi), forcing a workaround:
`uv pip install --no-deps -e .` to avoid `uv sync` downgrading torch, and
`.venv/bin/python` for the import test to avoid `uv run` triggering a re-sync.

**Isaac Sim 5.1.0.0** requires `torch==2.7.0` — compatible with our lockfile's
`torch==2.7.1`. This allows both Docker targets to share a single torch version
and simplifies the Dockerfile considerably.

## Changes

### `docker/Dockerfile`

| Change | File:Line | Reason |
|---|---|---|
| `isaacsim==6.0.1.0` → `isaacsim==5.1.0.0` | 62 | Compatible with torch 2.7.x |
| Removed `torch==2.11.0` upgrade line | — | No longer needed |
| Switched to `pip install` for isaacsim | 62-64 | uv's strict resolver rejects Windows-only deps (`pywin32`) and can't build `isaacsim-replicator` on Linux; `pip` is more lenient |
| Rollout `uv pip install --no-deps -e .` → `uv sync --frozen --no-dev` | 67 | Lockfile is now accurate for the whole env |
| Rollout `.venv/bin/python` → `uv run python` | 70 | No implicit sync conflict |
| Trainer consolidated sync + import test → single `uv run` | 49 | Cleaner, fewer layers |
| Comment header updated | 8 | Reflects new torch/isaacsim versions |

### Resulting structure

```
base (CUDA + apt + uv)
└── deps (uv sync, torch 2.7.1)
    ├── trainer (uv run, single .venv, torch 2.7.1)
    └── rollout (isaacsim 5.1.0, single .venv, torch 2.7.1)
```

Both targets use `uv run` natively. No torches to upgrade, no venvs to juggle.

## Tradeoff

Isaac Sim 5.1.0 (Oct 2025) vs 6.0.1 (Jun 2026) means we lose ~8 months of
upstream fixes and features. Worth revisiting if a need arises for 6.x.

## Files Changed

- `docker/Dockerfile` — see table above
- `docs/2026-07-22_05_switch-to-isaacsim-5.1.md` — this file
