# Phase A: Isaac Sim Environment + Data Generation

## Date
2026-07-23

## Overview
Implement the Isaac Sim pick-and-place environment, a scripted oracle for
demonstration generation, and wire property randomization. Assets are hosted
on a Hugging Face Storage Bucket and downloaded at setup time.

## Architectural decisions

| Question | Decision |
|----------|----------|
| Where does the task env live? | `episode_gen/sim/tasks/` — collocated with the simulation server |
| LeHome dependency? | None. probenet is self-contained. |
| Robot USD source? | Hugging Face Storage Bucket (`probenet/so101-assets`) |
| Simple object | Programmatic `UsdGeom.Box` prim — no USD asset needed |
| Shell object | Custom USD mesh in the HF bucket (future) |
| Camera | Static front view, 640×480 RGB (wrist cam optional later) |
| Oracle control | Joint-space interpolation (simpler than IK for baseline) |
| Training dependency on task env? | None — training consumes the LeRobot dataset, not the env |

## Directory layout (additions)

```
probenet/
├── assets/
│   └── robots/
│       └── .gitkeep                 # USD downloaded to here at runtime
├── episode_gen/sim/
│   ├── tasks/
│   │   ├── __init__.py              # gym.register("ProbeNet-SO101-PickPlace-Prim-v0")
│   │   ├── pick_place_prim_cfg.py   # PickPlacePrimEnvCfg
│   │   ├── pick_place_prim.py       # PickPlacePrimEnv (DirectRLEnv subclass)
│   │   ├── oracle.py                # PickPlaceStateMachine
│   │   └── mdp.py                   # Observation/reward/termination helpers
│   ├── server.py                    # Updated: _create_env() + --record mode
│   └── pyproject.toml               # Add isaaclab dependency
└── scripts/
    └── download_assets.sh           # Downloads USD assets from HF bucket
```

## A1 — Isaac Sim environment

### `pick_place_prim_cfg.py`

```python
@configclass
class PickPlacePrimSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = SO101_FOLLOWER_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )
    front: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/front_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, -0.5, 0.6),
            rot=(0.1650476, -0.9862856, 0.0, 0.0),
            convention="ros",
        ),
        data_types=["rgb"],
        spawn=PinholeCameraCfg(focal_length=28.7, ...),
        width=640, height=480,
        update_period=1/30.0,
    )
    light: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Light",
        spawn=DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

@configclass
class PickPlacePrimEnvCfg(DirectRLEnvCfg):
    scene: PickPlacePrimSceneCfg = PickPlacePrimSceneCfg(env_spacing=4.0)
    action_space = 6      # joint positions [pan, lift, elbow, wrist_flex, wrist_roll, gripper]
    observation_space = 6 # joint positions only (images handled separately)
    state_space = 0
    decimation = 1
    episode_length_s = 15.0
    action_scale = 1.0
```

### `pick_place_prim.py`

- Inherits `DirectRLEnv`
- `_setup_scene()`: creates robot articulation, camera, light, table (programmatic box), object (programmatic box/sphere)
- `_pre_physics_step()`: clones + scales actions
- `_apply_action()`: `self.scene["robot"].set_joint_position_target(self.actions)`
- `_get_observations()`: returns `{"policy": {"joint_pos": ..., "front": ...}}` matching what `IsaacSO101Env._pack_observation` expects
- `_get_rewards()`: `torch.zeros(self.num_envs)` — sparse, oracle provides supervision
- `_get_dones()`: timeout based on `max_episode_length`
- `_reset_idx()`: reset joint positions to default, zero velocities, randomize object position

### `__init__.py`

```python
gym.register(
    id="ProbeNet-SO101-PickPlace-Prim-v0",
    entry_point=f"{__name__}.pick_place_prim:PickPlacePrimEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pick_place_prim_cfg:PickPlacePrimEnvCfg",
    },
)
```

## A2 — Scripted pick-and-place oracle

### `oracle.py`

State machine with joint-space waypoints:

| Phase | Duration (steps) | Description |
|-------|-----------------|-------------|
| approach | 200 | Move gripper above the object |
| grasp | 50 | Close gripper |
| lift | 100 | Raise arm |
| place | 200 | Move to target position, open gripper |
| release | 100 | Retreat |

Joint targets are pre-computed offsets from the default rest pose. The
oracle records `(obs, action)` pairs for each step and exposes
`is_episode_done` / `check_success` / `reset` for the server loop.

## A3 — Wire property randomization

In `src/probenet/env/isaac_env.py:_randomize_properties()`, after sampling
values from `DEFAULT_PROPERTY_RANGES`, apply them to USD prims:

```python
from pxr import UsdPhysics

prim = self._env.scene.stage.GetPrimAtPath(object_prim_path)
mass_api = UsdPhysics.MassAPI.Apply(prim)
mass_api.GetMassAttr().Set(props["mass"])
```

Also wire friction via `PhysicsMaterialAPI` on the collision prim.

## A4 — Dataset generation

### Asset download (`scripts/download_assets.sh`)

Downloads the SO-101 USD from the HF Storage Bucket to `assets/robots/`:

```bash
# Downloads so101_follower.usd from the bucket to assets/robots/
```

### Server update (`episode_gen/sim/server.py`)

- `_create_env()` → `gym.make("ProbeNet-SO101-PickPlace-Prim-v0")`
- New `--record <path> --num-episodes <N>` flags
- In `--record` mode: after creating the env, runs the oracle for N episodes
  and saves (obs, action) pairs as a LeRobot-format dataset
- LeRobot dataset export uses lerobot's dataset writer (already a dependency)

### HF Storage Bucket

- Name: `probenet/so101-assets`
- Contents:
  - `robots/so101_follower.usd` — the articulated SO-101 robot USD
  - (future) `objects/shell.usd` — custom graspable shell mesh

## Implementation order

1. `scripts/download_assets.sh` — unblocks everything
2. Upload SO-101 USD to HF bucket
3. `episode_gen/sim/tasks/mdp.py` — observation/reward helpers
4. `episode_gen/sim/tasks/pick_place_prim_cfg.py`
5. `episode_gen/sim/tasks/pick_place_prim.py`
6. `episode_gen/sim/tasks/__init__.py`
7. `episode_gen/sim/tasks/oracle.py`
8. Update `episode_gen/sim/server.py` (_create_env + --record)
9. Wire property randomization in `isaac_env.py`
