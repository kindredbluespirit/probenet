# ProbeNet: Session Setup Report

**Date:** July 19 2026
**Author:** opencode (assistant)

This document records the full setup performed in the initial development
session so that future sessions can reconstruct context without re-screening
the entire codebase.

---

## 1. Project Foundation

### Repository structure

The project was already scaffolded with an empty `src/probenet/` package tree
(vision, probe, graph, policy, utils — all `__init__.py` stubs), a
`pyproject.toml` with no dependencies, CI/CD workflows, a Hugo project page
under `site/`, and a planning doc `docs/2026-7-16_01_init.md`.

### Python toolchain

- Package manager: **uv** (astral-sh/setup-uv)
- Build system: **hatchling** (`src/` layout)
- Python: >=3.12
- The `AGENTS.md` file specifies: use `uv add` for dependencies, `uv run` for
  scripts, and `uv run pytest` for tests.

---

## 2. Dependency installation

Initial `pyproject.toml` had empty `dependencies = []`. The following were
added via `uv add`:

| Package | Version | Notes |
|---|---|---|
| `mujoco` | >=3.10.0 | MuJoCo physics engine |
| `torch` | >=2.5.1 | PyTorch (CUDA 13.0) |
| `torchvision` | >=0.20.0,<0.27.0 | Constrained for lerobot compat |
| `gymnasium` | >=1.1.1 | Env interface |
| `hydra-core` | >=1.3.4 | Config framework |
| `lerobot` | >=0.6.0 | Training/dataset framework |

**Version gotcha:** lerobot 0.6.0 requires `gymnasium>=1.1.1` and
`torchvision>=0.22.0,<0.27.0`. The initial `uv add` installed torch 2.13 with
torchvision 0.28, which conflicts. The resolution was to constrain torchvision
to `<0.27.0` and gymnasium to `>=1.1.1`.

### Third-party assets

`mujoco_menagerie` is not a pip package — it's a collection of MJCF/STL files
on GitHub. It was cloned with `git clone --depth=1` into
`third_party/mujoco_menagerie/`. The `third_party/` directory was added to
`.gitignore`.

---

## 3. MuJoCo Environment (`probenet/env/`)

### Location
`src/probenet/env/so101_env.py`

### What it does
A `gymnasium.Env` subclass wrapping the Robot Studio SO-101 arm from MuJoCo
Menagerie (`robotstudio_so101/scene.xml`).

### Key design decisions

**Model loading.** The Menagerie scene uses relative includes (`so101.xml` with
`meshdir="assets"`). A composite XML string is written to a temp file next to
`scene.xml` in the Menagerie directory so that all mesh paths resolve
correctly. The composite XML adds a sphere object to the scene.

**Two shell objects.** Two presets with identical visual appearance (RGBA) but
different physics:

| Property | Shell A | Shell B |
|---|---|---|
| mass | 0.05 kg | 0.3 kg |
| friction | 1.2 | 0.3 |
| solref (compliance proxy) | -100 -10 (soft) | -1000 -100 (stiff) |
| RGBA | 0.6 0.35 0.25 1.0 | 0.6 0.35 0.25 1.0 |

**Object position.** Initial placement at `(0.25, 0, 0.035)` — the SO-101 arm
reach is ~0.3 m, so 0.5 m (from the Menagerie keyframe) was too far. The
object z is `radius = 0.035` so the sphere sits on the floor plane.

**Home pose.** `[0.0, -0.5, 0.8, 0.8, 1.58, 0.7]` in actuator ctrl space
(shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper).

**Action space.** Normalized `[-1, 1]` for 6 actuators, linearly mapped to the
actuator `ctrlrange` inside `step()`.

**Observation space.** Dict with:
- `image`: (224, 224, 3) uint8 RGB from a tracking camera (azimuth=135,
  elevation=-25, distance=0.6, trackbodyid=object).
- `state`: (25,) float32 — `qpos` (13) + `qvel` (12). The extra dims are the
  object freejoint (7 qpos + 6 qvel).

**Rendering.** Uses `mujoco.Renderer` with a tracking camera (not a named
camera in the XML). The camera follows the object via `mjCAMERA_TRACKING`.

**Probe signal access.** `env.get_probe_signal()` returns `actuator_force` and
`qfrc_actuator` arrays.

### Known quirks

- `mj_resetData` resets ctrl to zeros, so ctrl must be set *after* reset.
- The model is rebuilt from XML every time the env is created (~2 s). This is
  the bottleneck in data collection — reusing the model instance would be much
  faster.
- State dim 25 includes the object's freejoint. If you want just the arm state,
  slice `[7:13]` for qpos and `[6:12]` for qvel.

---

## 4. Probe Module (`probenet/probe/`)

### Location
`src/probenet/probe/probe_runner.py`

### What it does
Implements a scripted probe sequence (squeeze → lift-hold → release) and logs
actuator forces / joint torques.

### Probe phases

| Phase | Steps | Gripper |
|---|---|---|
| Approach | 200 | Open (0.0) |
| Squeeze | 100 | Closed (1.5) |
| Lift | 100 | Closed |
| Hold | 100 | Closed |
| Release | 100 | Open |

Joint targets during probe:
- Approach/squeeze: `[0.0, -0.5, 0.8, 0.8, 1.58, gripper]`
- Lift: `[0.0, -0.7, 0.8, 0.9, 1.58, 1.5]`

### Feature extraction
`extract_probe_features()` computes scalar statistics (mean, max, std) from the
actuator_force and qfrc_actuator arrays.

### Verification
The probe distinguishes Shell A from Shell B with 2517% mean-force contrast
(Shell A `af_mean=-0.021`, Shell B `af_mean=0.499`).

---

## 5. Conditioning Modules (`probenet/conditioning/`)

### Location
`src/probenet/conditioning/modules.py`

### Three variants

| Module | Inputs | Approach |
|---|---|---|
| `MLPConditioner` | vis + phys params | Concatenate → 2-layer MLP → 32-dim embedding |
| `GNNConditioner` | vis + phys params | Two nodes, message passing, concat → projection |
| `FiLMConditioner` | vis + phys params | Phys → scale/shift on vis features → projection |

All take `visual_dim=4` (shape, size, gloss, material) and `physical_dim=3`
(mass, compliance, friction). Factory function `build_conditioner(name)`.

---

## 6. Policy Module (`probenet/policies/`)

### Location
`src/probenet/policies/bc_policy.py`

### Components

- **`ImageEncoder`**: 4-layer CNN (16→32→64→64 channels, stride-2 convs) +
  adaptive avg pool + FC to 64-dim.
- **`StateEncoder`**: 2-layer MLP (25→64→16).
- **`BCPolicy`**: Image encoder + state encoder → concat (64+16=80) → 3-layer
  MLP head (128→64→6 actions). Optionally accepts a conditioning embedding
  (concatenated before the head).
- **`ProbeNetPolicy`**: Wraps BCPolicy with an external conditioner. If the
  conditioner is `None`, it's equivalent to the baseline.

---

## 7. Dataset Module (`probenet/dataset/`)

### Location
`src/probenet/dataset/sim_dataset.py`

### Two data access patterns

| Class | Approach | Use case |
|---|---|---|
| `ProbeNetDataset` | Loads all episodes into RAM at init | Fast random access (needs ~8 GB RAM for 100 episodes) |
| `EpisodicDataLoader` | Iterates one episode at a time | Low memory, but sequential (not random) access |

For the dataset with 100 episodes (55,000 frames, ~8 GB RGB), the
`ProbeNetDataset` was initially used but caused timeouts during training
because of memory pressure. The `EpisodicDataLoader` was written to work around
this: it shuffles episode order, loads one episode, iterates its frames in
batches, moves to the next episode. This is slower per epoch but avoids loading
all data at once.

### Serialization
- `save_episode()` writes `rgb.npy`, `state.npy`, `action.npy`, `metadata.json`
  per episode.
- `load_episode()` reads them back.
- `create_loaders()` splits episodes by episode index (not frame) to avoid
  data leakage.

### Metadata format
```json
{
    "object_type": "shell_a",
    "visual_params": {"shape": 1.0, "size": 0.035, "gloss": 0.5, "material": 0.0},
    "physical_params": {"mass": 0.05, "compliance": 1.0, "friction": 1.2},
    "probe_features": {"af_mean": ..., "af_max": ..., ...}
}
```

---

## 8. Scripts

### `scripts/collect_sim.py`
- Generates N episodes (default 100, split evenly across shells).
- For each episode: creates env, runs probe, runs scripted pick-and-place
  trajectory, saves data.
- Keyframes interpolated in joint space: home → approach → grasp → lift →
  carry → place → release → home (8 keyframes, ~630 steps total).
- Gaussian noise (σ=0.02) added to actions for variety.

### `scripts/train.py`
- Trains baseline BC, ProbeNet-MLP, or both.
- Uses `EpisodicDataLoader` for memory efficiency.
- Saves best checkpoint (by val loss) and training history JSON.
- 30 epochs of batch_size 128 takes ~10 minutes per variant on RTX 3060.

### `scripts/eval.py`
- Loads a checkpoint and runs N evaluation trials.
- Reports avg steps per shell.
- **Known limitation:** env has no termination, so always reports 500 steps.

### `scripts/verify_probe.py`
- Runs probe on both shells, renders images, extracts features, prints
  contrast ratio.

---

## 9. Key Gotchas and Lessons

1. **MuJoCo include resolution.** The composite XML must be written in the same
   directory as the included scene.xml, otherwise `meshdir="assets"` resolves
   incorrectly. Writing to `/tmp/` fails; writing to the Menagerie scene dir
   works.
2. **Action normalization.** The env maps actions from `[-1, 1]` to actuator
   `ctrlrange`. All scripted targets must be converted with
   `ctrl_to_action(ctrl, ctrl_range)`.
3. **State dimension.** `qpos (13) + qvel (12) = 25` includes the object
   freejoint. The `StateEncoder` defaults to `state_dim=25`.
4. **Data loading.** Loading all episodes into RAM (`np.concatenate`) causes
   OOM for >50 episodes on systems with limited RAM. The `EpisodicDataLoader`
   avoids this by loading one episode at a time.
5. **Object reach.** The SO-101 arm reach is ~0.3 m. The Menagerie keyframe
   places the object at (0.5, 0, 0.03), which is unreachable. The object was
   moved to (0.25, 0, 0.035).

## 10. Network / Generated files

During the setup, a composite XML file is written into the menagerie directory:
`third_party/mujoco_menagerie/robotstudio_so101/probenet_shell_a.xml` and
`probenet_shell_b.xml`. These are gitignored via `third_party/` but may need
manual cleanup if the directory layout changes.

The `outputs/` directory (gitignored) contains training checkpoints and
history JSONs. The `data/sim/` directory (gitignored) contains the 100-episode
dataset.

---

## 11. What Next

The immediate prioritized task list is in `docs/2026-07-19_01_status-and-roadmap.md`.

TL;DR:
1. Add reward/termination to the env for meaningful eval.
2. Implement the vision encoder.
3. Test GNN and FiLM conditioners.
4. Improve data generation with IK-based oracle.
5. Train the probe-based physical-parameter estimator.
6. Add ACT / Diffusion Policy backbones.
7. Real hardware integration.
