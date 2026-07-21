# ProbeNet: LeRobot + openpi Baseline Plan

**Date:** July 20 2026
**Status:** In progress — implementation

## Architecture Decision

- **Baseline policy:** π₀.₅ (openpi) fine-tuned on SO-101 pick-and-place demos
- **Data format:** LeRobotDataset v3 (MP4 + Parquet)
- **Sim env:** MuJoCo SO-101, adapted to LeRobot observation spec
- **Hardware:** SO-101 follower arm (Seeed Studio assembled kit) via LeRobot native SO101Follower class
- **Deployment:** openpi policy server (websocket) + probenet env client
- **No RECAP for baseline** — actions are simple pick-and-place; RECAP adds complexity for later

## Data Flow

```
┌─ Sim ───────────────────────────────────────────────┐
│  MuJoCo SO-101 → LerobotAdapter → obs dict           │
│    → record_sim.py → LeRobotDataset (HF Hub)         │
└──────────────────────────────────────────────────────┘
                          │
                          ▼
┌─ Training (openpi) ─────────────────────────────────┐
│  LeRobotDataset → openpi dataloader                   │
│    → π₀.₅ base checkpoint → fine-tune → ckpt         │
└──────────────────────────────────────────────────────┘
                          │
                          ▼
┌─ Eval ──────────────────────────────────────────────┐
│  MuJoCo env → LerobotAdapter → obs → websocket       │
│    → openpi policy server → action → env.step()      │
└──────────────────────────────────────────────────────┘

┌─ Hardware (future) ─────────────────────────────────┐
│  SO101Follower → lerobot-record → LeRobotDataset      │
│    → openpi fine-tune → policy server → hardware     │
└──────────────────────────────────────────────────────┘
```

## Observation Spec (unified sim + hardware)

Flat dict, matching LeRobot SO101Follower convention:

```python
obs = {
    "shoulder_pan.pos": 0.0,      # radians
    "shoulder_lift.pos": -0.5,
    "elbow_flex.pos": 0.8,
    "wrist_flex.pos": 0.8,
    "wrist_roll.pos": 1.58,
    "gripper.pos": 0.7,           # 0=open, 1=closed
    "cam_primary": ndarray,       # (224, 224, 3) uint8 HWC
}
```

Actions same format — absolute joint targets in radians.

## Repo Changes

### Remove (replaced by LeRobot + openpi)
- `src/probenet/policies/bc_policy.py`
- `src/probenet/dataset/sim_dataset.py`
- `scripts/collect_sim.py`
- `scripts/train.py`
- `scripts/eval.py`

### Add
- `src/probenet/env/lerobot_adapter.py`
- `scripts/record_sim.py`
- `scripts/eval_baseline.py`
- `configs/record_sim.yaml`
- `configs/so101_env.yaml`

### Modify
- `src/probenet/env/so101_env.py` — add LeRobot-compatible observation
- `pyproject.toml` — add transformers, scipy
- `tests/` — update for new observation spec
- `scripts/verify_probe.py` — update for new env interface

### Keep (unchanged)
- `src/probenet/probe/probe_runner.py`
- `src/probenet/conditioning/modules.py`
- `src/probenet/vision/` (placeholder)
- `src/probenet/graph/` (placeholder)
- `src/probenet/utils/paths.py`
- `site/`

### openpi (sibling repo) — Add
- `src/openpi/policies/so101_policy.py` — SO101Inputs / SO101Outputs
- `src/openpi/training/config.py` — `pi05_so101` TrainConfig entry

## Dependencies

```
probenet:
  lerobot >= 0.6.0, mujoco >= 3.10.0, torch >= 2.5.1,
  torchvision >= 0.20.0, <0.27.0, gymnasium >= 1.1.1,
  hydra-core >= 1.3.4, transformers >= 5.4.0, scipy >= 1.14.0

openpi (separate uv env, cloned as sibling):
  per its pyproject.toml
```
