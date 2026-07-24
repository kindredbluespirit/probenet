# ProbeNet: Implementation Plan (July 20, 2026)

## 1. Overview

ProbeNet extends VLA foundation models (π₀.₅ and GR00T N1.7) with a two-stage
interactive physical perception pipeline. Stage 1 (probing) estimates object
physical properties. Stage 2 (manipulation) conditions policy behavior on those
estimates.

The infrastructure separates training and rollout across rented cloud GPUs,
coordinated via HuggingFace Hub. Local hardware handles real-robot data
collection and inference.

---

## 2. Infrastructure

### Machine allocation

| Machine | Provider | GPU | Cost/hr | Role |
|---|---|---|---|---|
| Trainer | Lambda Labs | A100 80 GB | $1.29 | Fine-tune π₀.₅ / GR00T |
| Rollout | Lambda Labs | A10 24 GB | $0.50 | Isaac Sim data collection |
| Local | Own | RTX 3060 12 GB | $0 | Dev, real hardware, inference |

Lambda chosen as single provider: bare-metal instances (easier for Isaac Sim),
A100 cheaper than RunPod ($1.29 vs $1.39), native Isaac Sim support. A10 at
$0.50/hr handles Isaac Sim SO-101 scenes comfortably.

### Per-cycle cost estimate

| Task | Duration | GPU | Cost |
|---|---|---|---|
| Generate sim episodes | ~30 min | A10 | ~$0.25 |
| Compute norm stats | ~5 min | A100 | ~$0.11 |
| BC fine-tune π₀.₅ (LoRA) | ~2 hrs | A100 | ~$2.58 |
| Eval rollout | ~1 hr | A10 | ~$0.50 |
| **Total per training cycle** | | | **~$3.44** |

---

## 3. Architecture

```
┌─ Trainer Docker ──────────┐  ┌─ Rollout Docker ─────────┐  ┌─ Local ──────────────┐
│ Lambda A100                │  │ Lambda A10                │  │ RTX 3060              │
│ FROM nvidia/cuda:12.6      │  │ FROM nvidia/cuda:12.6     │  │ LeRobot teleop         │
│ Python 3.12 + uv           │  │ Python 3.12 + uv          │  │ SO-101 real arm        │
│ openpi + isaac-gr00t       │  │ openpi + isaac-gr00t      │  │ Record → HF Hub        │
│ torch 2.7.1, jax[cuda12]   │  │ + isaacsim[all]==6.0.1    │  │ Serve policy           │
│ ≈10 GB                     │  │ + torch 2.11.0            │  │                        │
│                            │  │ ≈30 GB                    │  │                        │
│ Train → upload ckpt        │  │ Isaac Sim SO-101 + obj    │  │                        │
│ ← download rollout data    │  │ Collect episodes          │  │                        │
└──────┬─────────────────────┘  │ Upload → HF Hub            │  └────────────────────────┘
       │                        └──────┬────────────────────┘
       │                               │
       └─────── HF Hub sync ───────────┘
        checkpoints + datasets + _complete markers
```

### Why two Docker images

Isaac Sim 6.0.1 requires `torch==2.11.0`. openpi pins `torch==2.7.1`. They
cannot coexist in one venv. Training image skips Isaac Sim. Rollout image
extends training image with Isaac Sim pip packages.

### Podman

Drop-in replacement for Docker. Same Dockerfile format, same OCI images.
Lambda instances run `install_podman.sh` on boot (podman + nvidia CDI config).

### Isaac Sim

Installed via pip, not a Docker base image:

```bash
uv pip install "isaacsim[all,extscache]==6.0.1" --extra-index-url https://pypi.nvidia.com
uv pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
```

No NGC base image overhead. Standard Python imports:

```python
import isaacsim
from isaacsim.simulation_app import SimulationApp
app = SimulationApp({"headless": True})
```

### Sync daemon

HF Hub is the message bus between trainer and rollout. No direct machine-to-machine
networking. Daemon runs as background thread in each container.

Key primitives:
- `get_latest_model_step()` — poll HF for newest checkpoint
- `download_latest_checkpoint()` — pull to versioned `step_N/` directory
- `list_available_rollouts()` — scan HF for new datasets (`_complete` marker check)
- `upload_rollout_dataset()` — push + atomic `_complete` marker
- Naming: `rollout_{step}_{strategy}_{timestamp}_{worker_id}`

Pattern adapted from lehome_solution's `hf_sync.py`.

---

## 4. Repo structure

```
probenet/
├── policies/
│   ├── openpi/              # git submodule → π₀.₅
│   └── isaac-gr00t/         # git submodule → GR00T N1.7
│
├── docker/
│   ├── Dockerfile           # multi-target: trainer + rollout
│   ├── docker-compose.yml   # local dev
│   └── install_podman.sh    # bootstrap Lambda instances
│
├── src/probenet/
│   ├── __init__.py
│   ├── cli.py               # entrypoint: --trainer | --rollout | --eval
│   ├── env/
│   │   ├── isaac_env.py     # Isaac Sim SO-101 wrapper (LeIsaac-based)
│   │   ├── lerobot_adapter.py
│   │   └── real_robot.py
│   ├── policies/
│   │   ├── so101.py         # shared obs/action spec
│   │   ├── pi05.py          # π₀.₅ Inputs/Outputs + TrainConfig
│   │   └── gr00t.py         # GR00T EmbodimentTag + TrainerConfig
│   ├── probe/               # scripted probe, signal extraction
│   ├── conditioning/        # MLP/GNN/FiLM modules
│   ├── training/
│   │   ├── trainer.py       # policy-agnostic: --policy pi05|gr00t
│   │   └── config.py
│   ├── rollout/
│   │   └── worker.py        # poll ckpt → collect → upload loop
│   └── sync/
│       ├── hub.py           # HF Hub primitives
│       └── daemon.py        # background async thread
│
├── framework/               # reusable across future projects
│   ├── sync/                # identical HF Hub sync code
│   ├── modes/               # generic trainer/rollout/eval loop templates
│   └── config.py            # shared config schema
│
├── configs/
│   ├── trainer.yaml
│   ├── rollout.yaml
│   └── eval.yaml
│
├── scripts/
├── tests/
├── pyproject.toml
├── .github/workflows/
│   └── docker-publish.yml
└── docs/
```

### Removed

- `src/probenet/env/so101_env.py` — MuJoCo env
- `src/probenet/policies/bc_policy.py` — old BC policy
- `src/probenet/dataset/` — LeRobot handles datasets
- `third_party/mujoco_menagerie/` — MuJoCo assets
- `mujoco` dependency in `pyproject.toml`

### Kept

- `probe/` — scripted probe runner (core to ProbeNet)
- `conditioning/` — MLP/GNN/FiLM (core contribution)
- `vision/` — placeholder for future vision encoder
- `graph/` — placeholder for future GNN property translator

---

## 5. Isaac Sim — via LeIsaac

LeIsaac (697 stars, Apache 2.0, `LightwheelAI/leisaac`) provides:

| Component | Source |
|---|---|
| SO-101 USD robot model | `assets/robots/so101_follower.usd` |
| Scene loading | IsaacLab scene configs |
| Scripted data generation | `datagen/state_machine/` module |
| LeRobot dataset conversion | Built-in HDF5 → LeRobot recorder |
| IsaacLab task templates | DirectEnv with cameras, observations, terminations |

LeIsaac uses IsaacLab as a dependency (`isaaclab` pip package). ProbeNet builds
its custom SO-101 probe environment on top: adding the scripted probe sequence,
actuator force logging, property randomization, and LeRobot-compatible
observation output.

We do **not** need the official Isaac Sim assets pack (`~/isaacsim_assets`).
LeIsaac's USD files are self-contained.

### Version compatibility

LeIsaac currently tested on IsaacSim 5.1 (Python 3.11, torch 2.7.0, CUDA 12.8).
IsaacSim 6.0.1 compatibility to be verified. Fallback: pin IsaacSim to 5.1.

---

## 6. Policy backends

### π₀.₅ (Physical Intelligence)

- Backend: JAX (primary), PyTorch (inference)
- Training: `scripts/train.py` in openpi repo
- LoRA fine-tune: >22.5 GB VRAM
- Works on Lambda A100 80 GB
- Dependency pinned: `torch==2.7.1`, `jax[cuda12]==0.5.3`, `flax==0.10.2`

### GR00T N1.7 (NVIDIA)

- Backend: PyTorch
- Training: `gr00t/experiment/launch_finetune.py`
- Has official SO-101 fine-tuning tutorial
- Native `EmbodimentTag` system for custom robot configs
- Single GPU fine-tuning supported
- 3B parameters

### CLI

```bash
podman run probenet --trainer --policy pi05
podman run probenet --trainer --policy gr00t
podman run probenet --rollout
podman run probenet --eval --policy pi05 --ckpt /data/step_5000
```

---

## 7. Research plan

Two baselines, then ProbeNet conditioning on top of both:

| | Baseline | ProbeNet |
|---|---|---|
| **π₀.₅** | BC pick-and-place, no property info | + physical property conditioning |
| **GR00T N1.7** | BC pick-and-place, no property info | + physical property conditioning |

The data pipeline and observation spec are **policy-agnostic**. Both policies
consume the same LeRobotDataset format. Property conditioning tokens are text
tokens injected into the prompt — same approach for π₀.₅'s multimodal prefix
and GR00T's language prompt.

This makes the ablation study model-agnostic: if ProbeNet improves both
backbones, the contribution is general.

---

## 8. Data flow

```
Phase 1: Scripted oracle (BC only)
┌─ Rollout (A10) ──────────────────────────────────────┐
│ Isaac Sim SO-101 + objects with known properties      │
│ Scripted oracle: squeeze → lift → place               │
│ Export LeRobotDataset → upload to HF Hub              │
└──────────────────────────────────────────────────────┘
                         │
                         ▼
┌─ Trainer (A100) ─────────────────────────────────────┐
│ Download dataset from HF Hub                          │
│ Compute norm stats                                    │
│ BC fine-tune (LoRA) → upload checkpoint               │
└──────────────────────────────────────────────────────┘

Phase 2+: Online RL (sync daemon active)
┌─ Rollout ────────────────────────────────────────────┐
│ poll ckpt → collect episodes → upload → repeat       │
└──────────────────────────────────────────────────────┘
         │
         ▼
┌─ Trainer ────────────────────────────────────────────┐
│ poll datasets → download → recompute adv → RL train  │
│ → upload ckpt → repeat                               │
└──────────────────────────────────────────────────────┘

Real hardware (Phase 9+):
┌─ Local ──────────────────────────────────────────────┐
│ LeRobot teleop → record real demos → upload to HF Hub │
│ Download ckpt → serve policy → control real arm       │
└──────────────────────────────────────────────────────┘
```

---

## 9. Implementation phases

| # | Phase | Key Deliverables |
|---|---|---|
| 1 | **Docker + policies** | Dockerfile (trainer + rollout targets), `policies/openpi/`, `policies/gr00t/`, `pyproject.toml` merged |
| 2 | **Isaac Sim env** | `env/isaac_env.py` via LeIsaac, `env/lerobot_adapter.py`, verify IsaacSim version compat |
| 3 | **Data pipeline** | Shared obs/action spec, scripted oracle data collection, LeRobotDataset export, HF Hub upload |
| 4 | **π₀.₅ baseline** | `policies/pi05.py`, BC fine-tune, eval metrics |
| 5 | **GR00T baseline** | `policies/gr00t.py`, BC fine-tune, eval metrics |
| 6 | **Sync daemon** | `sync/hub.py`, `sync/daemon.py`, HF Hub coordinator |
| 7 | **Rollout worker** | `rollout/worker.py`, auto poll→collect→upload loop |
| 8 | **ProbeNet π₀.₅** | Property conditioning tokens added to π₀.₅ prompt, modality dropout |
| 9 | **ProbeNet GR00T** | Property conditioning tokens added to GR00T prompt |
| 10 | **Eval pipeline** | Success/failure detection, per-object metrics, comparison tables |
| 11 | **Real hardware** | LeRobot teleop recording, policy server on local GPU, SO-101 control |
| 12 | **Probe Encoder** | CNN + Transformer over raw probe signal, supervised param estimation |
| 13 | **GNN Translator** | Bidirectional property graph, latent ↔ interpretable mapping |
| 14 | **Learned probing** | RECAP-style advantage conditioning, autonomous probe policy |
| 15 | **Aggressiveness** | Continuous steering parameter, graded behavior in probe + manipulation |

---

## 10. Configuration

### `configs/trainer.yaml`

```yaml
hf_model_repo: "<user>/probenet-model"
hf_dataset_repo: "<user>/probenet-dataset"
policy: pi05           # pi05 | gr00t
batch_size: 64
lr: 5e-5
num_steps: 100000
save_interval: 5000
wandb_enabled: true
init_checkpoint: "gs://openpi-assets/checkpoints/pi05_base"
```

### `configs/rollout.yaml`

```yaml
hf_model_repo: "<user>/probenet-model"
hf_dataset_repo: "<user>/probenet-dataset"
num_episodes: 100
num_workers: 4
headless: true
property_randomization:
  mass: [0.01, 5.0]
  friction: [0.0, 2.0]
  compliance: [0.0, 1.0]
```

---

## 11. Key dependencies

### Trainer image (`torch==2.7.1`)

```
openpi: jax[cuda12]==0.5.3, flax==0.10.2, torch==2.7.1, orbax-checkpoint==0.11.13
isaac-gr00t: torch==2.7.1, flash-attn (TBD)
probenet: lerobot>=0.6.0, hydra-core>=1.3.4, wandb, huggingface-hub[cli]
```

### Rollout image (`torch==2.11.0`)

```
All of the above (except jax/flax — not needed for collecting data)
+ isaacsim[all,extscache]==6.0.1
+ isaaclab
+ leisaac (git+https://github.com/LightwheelAI/leisaac.git)
```

### pyproject.toml sources

```toml
[tool.uv.sources]
openpi = { path = "policies/openpi", editable = true }
isaac-gr00t = { path = "policies/gr00t", editable = true }
```

---

## 12. References & prior art

| Project | What we learned |
|---|---|
| **lehome_solution** (IliaLarchenko) | Disaggregated trainer/rollout, HF Hub sync daemon, Isaac Sim rollout workers |
| **openpi** (Physical Intelligence) | π₀.₅ model, policy server/client via websocket, LeRobot fine-tuning pipeline |
| **LeIsaac** (LightwheelAI) | SO-101 IsaacLab integration, state machine `datagen`, LeRobot conversion |
| **π_RL** (Tsinghua, 2025) | PPO on flow-based VLAs, 320 parallel sim envs, flow-matching RL algorithms |
| **HIL-SERL** (Berkeley, 2024) | Real-robot RL with human interventions, SAC + reward classifier, actor/learner split |
| **RLinf-VLA** (Tsinghua, 2025) | Disaggregated + hybrid GPU allocation for VLA+RL, 1.61–1.88× speedup |
| **GR00T N1.7** (NVIDIA, 2026) | SO-101 fine-tuning tutorial, `EmbodimentTag` system, LeRobot dataset support |
| **Isaac Lab** (NVIDIA) | Container deployment, Docker-based Isaac Sim, pip installation support |

---

## 13. Open questions

1. **LeIsaac + IsaacSim 6.0.1 compatibility?** LeIsaac tested on 5.1. Verify
   6.0.1 or pin to 5.1.
2. **ProbeNet custom objects?** Need shell/cup/convex USD meshes for probing
   tasks. Create simple geometric shapes or adapt from LeIsaac object pipeline.
3. **Real hardware status?** Is physical SO-101 assembled? Affects Phase 11
   priority.
4. **Phase 1 dataset size?** Architecture plan says 5,000–10,000 episodes. Start
   with 100–500 for initial baseline, scale up for RL phases.
5. **GR00T dependency details?** Need to verify exact `isaac-gr00t` pip install
   compatibility with our Docker image specs.
6. **Wandb vs other logging?** Wandb is the default (both openpi and lehome use
   it). Free for academic use.
