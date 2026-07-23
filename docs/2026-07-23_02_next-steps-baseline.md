# Next Steps: Baseline openpi Finetune + Rollout

## Goal
Get the vanilla π₀.₅ baseline working end-to-end: generate a dataset in Isaac Sim,
fine-tune on it, and roll out the trained policy.

## Phase A: Isaac Sim environment

### A1. Unblock unbuntu: `episode_gen/sim/server.py:_create_env()`
The WebSocket server exists as a skeleton. The blocking work is creating
the actual LeIsaac task environment:

```python
# episode_gen/sim/server.py  →  IsaacSimServer._create_env()
```

This needs a `SingleArmTaskDirectEnv` subclass (following lehome-challenge's
`GarmentEnv` pattern in `source/lehome/lehome/tasks/bedroom/garment_bi_v2.py`).

Minimal env:
- SO-101 robot (URDF/USD from lerobot) on a table
- One camera (front view, 480×640)
- One graspable object (shell) at a random position
- `_get_observations()` returns `joint_pos` + `front` camera
- `_apply_action()` sets target joint positions
- Reward: distance to target (optional for BC)
- Done: gripper reaches target height

### A2. Scripted pick-and-place oracle
A state machine that generates demonstration trajectories:

```
approach (pre-grasp pose above object)
  → grasp (close gripper)
  → lift (raise arm)
  → place (move to target, open gripper)
  → release (retreat)
```

The `ProbeRunner` in `src/probenet/probe/probe_runner.py` already has the
phase-machine pattern — use it as a template.

### A3. Wire property randomization
`IsaacSO101Env._randomize_properties()` samples mass/friction/compliance but
never applies them to USD prims. Needs:
```python
prim = self._stage.GetPrimAtPath(object_usd_path)
mass_api = UsdPhysics.MassAPI(prim)  # or set mass attribute directly
mass_api.GetMassAttr().Set(random_mass)
```

### A4. Generate initial dataset
```bash
cd episode_gen/sim
uv sync
uv run python server.py --headless --record data/lerobot/so101_pick_place --num-episodes 100
```

This produces a LeRobot dataset at `data/lerobot/so101_pick_place/`.

## Phase B: openpi training pipeline

### B1. Register `pi05_so101` config
Add a config entry to `src/probenet/training/config.py` (following lehome's
pattern in `lehome_solution/training/config.py`):

```python
_CONFIGS = [
    TrainConfig(
        name="pi05_so101",
        model=Pi0Config(pi05=True, action_dim=6, action_horizon=50),
        data=SimpleDataConfig(
            repo_id="so101_pick_place",
            asset_id="so101",
            repack_transforms=...,
            data_transforms=Group(inputs=[So101Inputs()]),
        ),
        weight_loader=CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
    ),
]
```

### B2. Compute norm stats
```bash
uv run python scripts/compute_norm_stats.py \
  --dataset data/lerobot/so101_pick_place \
  --asset-id so101 --output-dir outputs/assets
```

### B3. Run training
```bash
uv sync --extra training
uv run python scripts/train.py \
  --config-name pi05_so101 \
  --exp-name baseline_v1 \
  --dataset data/lerobot/so101_pick_place
```

This calls openpi's JAX training loop. Checkpoints land at
`outputs/checkpoints/pi05_so101/baseline_v1/`.

## Phase C: Inference + Rollout

### C1. Start inference server
```bash
cd inference/openpi
uv sync
uv run python serve.py \
  --checkpoint-dir ../../outputs/checkpoints/pi05_so101/baseline_v1/50000 \
  --port 8000
```

### C2. Start Isaac Sim server
```bash
cd episode_gen/sim
uv run python server.py --port 8226
```

### C3. Run orchestrator
```bash
uv run python scripts/rollout.py \
  --policy openpi --policy-url ws://localhost:8000 \
  --data-gen sim --data-gen-url ws://localhost:8226 \
  --num-episodes 50 \
  --output-dir outputs/eval_runs/baseline_v1
```

## Phase D: Real hardware (RTX 3060)

### D1. Tune Isaac Sim cameras to match real setup
- Same FOV, resolution, mounting position
- Same lighting conditions
- Same background color/texture

### D2. Implement `episode_gen/so101/server.py:_connect_robot()`
Uses lerobot's `SO101Follower` for robot control and `SO101Leader` for
DAgger teleop:
```python
from lerobot.common.robots.so101 import SO101Follower, SO101Leader
```

### D3. Run real robot rollout
```bash
cd episode_gen/so101
uv sync
uv run python server.py --port 8227 --port-config ../../configs/real_robot.yaml
```

Then from root:
```bash
uv run python scripts/rollout.py \
  --policy openpi --policy-url ws://localhost:8000 \
  --data-gen so101 --data-gen-url ws://localhost:8227
```

## Files that need changes

| File | Change |
|------|--------|
| `episode_gen/sim/server.py` | Implement `_create_env()` with LeIsaac task |
| `src/probenet/training/config.py` | Add `pi05_so101` to `_CONFIGS` |
| `scripts/compute_norm_stats.py` | Create (doesn't exist yet) |
| `src/probenet/env/isaac_env.py` | Wire `_randomize_properties()` to USD |
| `episode_gen/so101/server.py` | Implement `_connect_robot()` (Phase D) |
