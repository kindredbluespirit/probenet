# ProbeNet Architecture: Multi-Service Design

## Date
2026-07-23

## Overview

ProbeNet is organized as a **multi-service system** with separate venvs for each
component. Services communicate over TCP/WebSocket (or ZeroMQ for GR00T), making
them independently deployable across machines or collocated on localhost.

## Directory layout

```
probenet/
├── pyproject.toml              # Orchestrator + training scripts (light venv)
├── src/probenet/
│   ├── clients/                # PolicyClient + DataGenClient abstractions
│   ├── policies/               # Config registry, transforms (So101Inputs, etc.)
│   └── training/               # Training config, trainer orchestration
│
├── scripts/
│   ├── train.py                # Training orchestrator (policy-agnostic)
│   └── rollout.py              # Episode loop orchestrator (policy + data-gen agnostic)
│
├── policies/                  # Policy inference servers
│   ├── openpi/                 #   π₀.₅ server (WebSocket)
│   │   ├── pyproject.toml      #   jax, flax, openpi (editable from policies/)
│   │   └── serve.py            #   loads checkpoint, serves WebSocket
│   └── gr00t/                  #   GR00T server (ZeroMQ) — future
│       ├── pyproject.toml
│       └── serve.py
│
├── episode_gen/                # Data generators
│   ├── sim/                    #   Isaac Sim data generator
│   │   ├── pyproject.toml      #   isaacsim, isaaclab, lerobot
│   │   └── server.py           #   WebSocket server: reset, step, get_obs
│   └── so101/                  #   Real SO-101 robot data generator
│       ├── pyproject.toml      #   lerobot, pyserial, pyrealsense2
│       └── server.py           #   WebSocket server + DAgger teleop mode
│
├── policies/                   # Git submodules (actual model repos + servers)
│   ├── openpi/                 #   Physical Intelligence openpi
│   └── isaac-gr00t/            #   NVIDIA GR00T N1.7
│
├── configs/                    # YAML configs for training/rollout
├── data/                       # Datasets (LeRobot format)
└── outputs/                    # Checkpoints, eval results, logs
```

## The five venvs

| # | Directory | `uv sync` | Purpose | Key dependencies |
|---|-----------|-----------|---------|-----------------|
| 1 | `probenet/` (root) | `uv sync` | Orchestrator + training | websockets, numpy, hf-hub, pyyaml |
| 2 | `policies/openpi/` | `cd policies/openpi && uv sync` | π₀.₅ inference | jax, flax, openpi, orbax |
| 3 | `policies/gr00t/` | `cd policies/gr00t && uv sync` | GR00T inference | torch, gr00t, pyzmq |
| 4 | `episode_gen/sim/` | `cd episode_gen/sim && uv sync` | Isaac Sim data gen | isaacsim, isaaclab, lerobot |
| 5 | `episode_gen/so101/` | `cd episode_gen/so101 && uv sync` | Real robot data gen | lerobot, pyserial, pyrealsense2 |

For training, the root venv can optionally pull in openpi deps:
```bash
uv sync --extra training   # adds openpi + jax + flax + lerobot
```

## Wire protocols

### Inference → orchestrator

| Backend | Protocol | Port | Library |
|---------|----------|------|---------|
| openpi | WebSocket | 8000 | `websockets` (built into openpi) |
| GR00T | ZeroMQ | 5555 | `pyzmq` (built into GR00T) |

Message format (request → response):
```
→ {"type": "infer", "obs": {joints, images, ...}}
← {"actions": [...], "metadata": {...}}
```

### Episode gen → orchestrator

Both sim and so101 use the same WebSocket protocol:

```
→ {"type": "reset", "seed": 42}
← {"obs": {joints, images, ...}, "info": {...}}

→ {"type": "step", "action": [0.1, 0.2, ...]}
← {"obs": {joints, images, ...}, "reward": 0.0, "done": false}
```

## Abstraction layer

`src/probenet/clients/` provides thin protocol wrappers:

```python
class PolicyClient:
    """Abstract inference client."""
    def infer(self, obs: dict) -> dict: ...

class WebSocketPolicyClient(PolicyClient):   # → policies/openpi/
class ZMQPolicyClient(PolicyClient):          # → policies/gr00t/

class DataGenClient:
    """Abstract data generator client."""
    def reset(self, seed: int = 0) -> dict: ...
    def step(self, action: np.ndarray) -> tuple[dict, float, bool]: ...

class SimClient(DataGenClient):              # → episode_gen/sim/
class RealRobotClient(DataGenClient):         # → episode_gen/so101/
```

## Episode loop (orchestrator)

```python
# scripts/rollout.py
policy = WebSocketPolicyClient("ws://localhost:8000")
data_gen = SimClient("ws://localhost:8226")

while episodes_collected < target:
    obs = data_gen.reset()
    done = False
    while not done:
        result = policy.infer(obs)
        obs, reward, done = data_gen.step(result["actions"])
        dataset.add_frame(obs, result["actions"])
    dataset.save_episode()
    episodes_collected += 1
```

## Environment segregation rules

### `episode_gen/` owns
- Creating the env (Isaac Sim scene or real robot connection)
- Stepping physics / reading sensors (sim) or reading hardware (real)
- Scripted oracle / state machine for autonomous data collection
- Applying actions to motors
- Saving episodes as LeRobot datasets
- Property randomization (mass, friction, compliance)

### `probenet/` (orchestrator) owns
- Connecting to servers (sends messages, no env imports)
- The episode loop (reset → obs → infer → act → repeat)
- HF Hub sync (upload/download datasets + checkpoints)
- Training pipeline (loads datasets from disk, trains policy)

### `policies/` owns
- Loading model checkpoints
- Running model inference (obs → actions)
- Returning actions over wire protocol

### No overlap
- probenet never imports `isaacsim`, `lerobot`, or any env code
- probenet never creates a dataset or touches parquet files
- episode_gen never imports `openpi`, `gr00t`, `jax`, or any model code
- episode_gen doesn't know about HF Hub — just writes files to local path
- inference never imports simulation or robot code

## Deployment scenarios

### Local RTX 3060 (real hardware)
```
probenet (orchestrator)  →  policies/openpi (model, GPU)
                         →  episode_gen/so101 (real robot, leader arms)
```

### Cloud A10/A100 (Isaac Sim)
```
probenet (orchestrator)  →  policies/openpi (model, GPU)
                         →  episode_gen/sim (Isaac Sim)
```

### Mixed (train on A100, rollout on A10)
```
probenet (orchestrator, A10)
  →  policies/openpi (A100, LAN WebSocket)
  →  episode_gen/sim (A10, localhost WebSocket)

HF Hub sync daemon:
  A100 trainer uploads checkpoints → HF Hub
  A10 rollout worker downloads checkpoints from HF Hub
```

## Phase plan

| Phase | What | Depends on |
|-------|------|-----------|
| A | Isaac Sim env + scripted oracle + dataset gen | Isaac Sim runtime |
| B | openpi config registry + training pipeline | Phase A (dataset) |
| C | Inference server + orchestrator rollout loop | Phase B (checkpoint) |
| D | Real robot adapter + camera tuning | Phase C (can start in parallel) |
| E | GR00T backend | Phase B-C |
| F | ProbeNet conditioning (property tokens) | Phase B-C |
