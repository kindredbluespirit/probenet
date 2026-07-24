# Phase A Implementation: Isaac Sim Environment + Data Generation

## Date
2026-07-23

## What was built

Phase A delivers a minimal Isaac Sim pick-and-place environment (prim-based),
a scripted oracle for generating demonstration trajectories, property
randomization wired to USD prims, and the infrastructure to collect datasets.

## Architecture decisions

| Decision | Choice |
|----------|--------|
| Task env location | `episode_gen/sim/tasks/` — collocated with the simulation server |
| LeHome dependency | None — probenet is self-contained |
| Robot USD source | Hugging Face Storage Bucket `kindredbluespirit/so101-assets` |
| Simple object | Programmatic `UsdGeom.Box` prim (no USD asset needed) |
| Camera | Static front view, 640×480 RGB |
| Oracle control | Joint-space interpolation (approach → grasp → lift → place → release) |
| Property randomization | `UsdPhysics.MassAPI.Apply()` on the prim |

## Files created

| File | Purpose |
|------|---------|
| `episode_gen/sim/tasks/__init__.py` | Gym registration: `ProbeNet-SO101-PickPlace-Prim-v0` |
| `episode_gen/sim/tasks/pick_place_prim_cfg.py` | `PickPlacePrimEnvCfg` (robot, camera, object, sim params) |
| `episode_gen/sim/tasks/pick_place_prim.py` | `PickPlacePrimEnv(DirectRLEnv)` — scene setup, stepping, reset |
| `episode_gen/sim/tasks/robot_cfg.py` | `SO101_FOLLOWER_CFG` articulation config + joint limits |
| `episode_gen/sim/tasks/oracle.py` | `PickPlaceStateMachine` — 5-phase pick-and-place |
| `episode_gen/sim/tasks/mdp.py` | Re-exports for Isaac Lab MDP observation terms |
| `scripts/download_assets.sh` | Downloads SO-101 USD from `kindredbluespirit/so101-assets` |
| `src/probenet/utils/assets.py` | `ASSETS_ROOT` resolver (env var `PROBENET_ASSETS_ROOT` or git root) |

## Files modified

| File | Change |
|------|--------|
| `episode_gen/sim/server.py` | Implemented `_create_env()` via `gym.make`, added `--record`/`--num-episodes`/`--task` flags, added `_run_record_mode()` |
| `episode_gen/sim/pyproject.toml` | Added `isaaclab[isaacsim,all]>=2.3.0` dependency |
| `src/probenet/env/isaac_env.py` | `_randomize_properties()` now applies mass to USD prims via `UsdPhysics.MassAPI` |

## HF Storage Bucket

- **URL**: `https://huggingface.co/buckets/kindredbluespirit/so101-assets`
- **Contents**: SO-101 articulated robot USD with mesh + physics data (~23 MB)
- **Path in bucket**: `robots/so101_new_calib/so101_new_calib.usd` (references sublayers in `configuration/`)
- **Download**: `bash scripts/download_assets.sh` (uses `hf sync` or `huggingface_hub`)

## How to test

### Prerequisites

```bash
# 1. Activate the Isaac Sim venv
cd episode_gen/sim
uv sync

# 2. Download the SO-101 robot USD
bash ../../scripts/download_assets.sh
```

### Test the environment directly

```bash
cd episode_gen/sim
uv run python -c "
import gymnasium as gym
import tasks  # registers the env
import torch
from isaaclab_tasks.utils import parse_env_cfg

cfg = parse_env_cfg('ProbeNet-SO101-PickPlace-Prim-v0', device='cuda:0', num_envs=1)
env = gym.make('ProbeNet-SO101-PickPlace-Prim-v0', cfg=cfg).unwrapped
obs, _ = env.reset()
print('Observation keys:', obs.keys())
print('Joint pos:', obs['policy']['joint_pos'])
env.close()
"
```

### Generate a dataset (headless)

```bash
cd episode_gen/sim
uv run python server.py \
  --headless \
  --record ../../data/lerobot/so101_pick_place \
  --num-episodes 50 \
  --task ProbeNet-SO101-PickPlace-Prim-v0
```

This runs the oracle for 50 episodes and saves NPZ files to
`data/lerobot/so101_pick_place/episode_*.npz`, each containing
`observations` and `actions` arrays.

### Run the WebSocket server (interactive)

```bash
cd episode_gen/sim
uv run python server.py --port 8226 --headless
```

Connect with a WebSocket client:

```json
→ {"type": "reset", "seed": 42}
← {"status": "ok", "obs": {...}, "info": {...}}

→ {"type": "step", "action": [0.0, -0.5, 0.8, 0.8, 1.58, 0.0]}
← {"status": "ok", "obs": {...}, "reward": 0.0, "done": false, "info": {...}}
```

## File-by-file walkthrough

### `pick_place_prim_cfg.py`

Config class with:
- `PickPlacePrimEnvCfg(DirectRLEnvCfg)` with `action_space=6`, `episode_length_s=15`
- `robot`: SO-101 follower articulation at `/World/Robot`
- `object`: RigidObject box (4 cm cube, mass 0.05 kg) at a fixed position
- `front_camera`: 640×480 RGB camera on the robot base
- Physics: 90 Hz, 1 decimation

### `pick_place_prim.py`

Env class with:
- `_setup_scene()`: creates Articulation, RigidObject, TiledCamera, DomeLight
- `_pre_physics_step()`: clones actions, applies `action_scale`
- `_apply_action()`: `set_joint_position_target()`
- `_get_observations()`: returns `{"policy": {"joint_pos": ..., "actions": ...}}`
- `_get_rewards()`: returns `torch.zeros()` (sparse — oracle provides supervision)
- `_get_dones()`: time-out based on `max_episode_length`
- `_reset_idx()`: writes default joint positions, zeroes velocities

### `robot_cfg.py`

Self-contained SO-101 articulation config:
- USD path: `assets/robots/so101_new_calib/so101_new_calib.usd`
- 6 joints: `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper`
- Two actuator groups: `sts3215-arm` (5 DOF) and `sts3215-gripper` (1 DOF)
- Joint limits, motor limits, and rest pose ranges documented

### `oracle.py`

State machine with 5 phases:
| Phase | Steps | Action |
|-------|-------|--------|
| approach | 200 | Move arm above object position |
| grasp | 50 | Close gripper |
| lift | 100 | Raise arm |
| place | 200 | Move arm to target, open gripper |
| release | 100 | Retreat to rest pose |

Implements the `StateMachineBase` interface: `setup()`, `get_action()`,
`advance()`, `check_success()`, `reset()`, `is_episode_done`.

### `server.py`

Two modes:
- **WebSocket mode** (default): serves `reset`, `step`, `get_obs`, `close` endpoints
- **Record mode** (`--record`): creates env, runs oracle for N episodes, saves NPZs

## Next steps (Phase B)

The generated dataset feeds into the openpi training pipeline:
1. Register a `pi05_so101` config in `src/probenet/training/config.py`
2. Compute norm stats
3. Run `scripts/train.py`
4. Start inference server + roll out
