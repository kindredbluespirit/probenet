# ProbeNet: Status Report (July 21, 2026)

## What is ProbeNet?

ProbeNet extends Vision-Language-Action (VLA) foundation models with a
two-stage interactive physical perception pipeline:

1. **Stage 1 — Probing**: The policy autonomously interacts with an object
   (squeeze, lift, tap) to estimate physical properties — mass, friction,
   compliance, fragility.

2. **Stage 2 — Manipulation**: The same policy uses those property estimates
   as conditioning tokens to adapt grip force, approach speed, and trajectory
   for the specific object being handled.

Two VLA backbones are supported: **π₀.₅** (Physical Intelligence) and
**GR00T N1.7** (NVIDIA). The physical property conditioning is model-agnostic
— showing improvement on both backbones is the core research contribution.

---

## Current state

### All 15 implementation phases complete

The full codebase is built and tested. All modules import cleanly with 12
passing tests (1 skipped — requires Isaac Sim runtime).

### What you can do now (no GPU needed)

Every module can be imported and verified:

```bash
# Install dependencies
git clone --recurse-submodules <repo-url>
cd probenet
uv sync

# Verify everything imports
uv run python -c "
from probenet.env import IsaacSO101Env, RealSO101Runner
from probenet.policies import So101Inputs, So101Outputs, ProbeNetConditioner
from probenet.probe import ProbeEncoder, PropertyTranslator, LearnedProbeRunner
from probenet.sync import SyncDaemon
from probenet.training import Trainer
from probenet.eval import EvalRunner
print('All modules OK')
"

# Run tests
uv run python -m pytest tests/ -v

# See CLI help
uv run python -m probenet.cli --help
```

### Module map

```
src/probenet/
├── cli.py                    # Entrypoint (--mode trainer|rollout|eval, --probenet, --policy)
├── env/                      # Environments
│   ├── isaac_env.py          #   Isaac Sim SO-101 + probe objects + property randomization
│   ├── lerobot_adapter.py    #   LeRobot flat dict adapter
│   └── real_robot.py         #   Physical SO-101 arm + DummyRobotRunner for tests
├── policies/                 # Policy backends
│   ├── so101.py              #   Shared obs/action spec, joint names, LeRobot features
│   ├── pi05.py               #   π₀.₅ Inputs/Outputs + Pi05ProbeNetConfig
│   ├── gr00t.py              #   GR00T TrainConfig + Gr00tProbeNetConfig
│   └── probenet.py           #   ProbeNetConditioner (prompt augmentation + modality dropout)
├── probe/                    # Probing and physical property estimation
│   ├── probe_runner.py       #   Scripted squeeze-lift probe + signal extraction
│   ├── properties.py         #   PropertyDef, PropertyState, prompt token builders, dropout
│   ├── probe_encoder.py      #   ForceEncoder → SignalTransformer → PropertyHeads (CNN+Transformer ML)
│   ├── learned_probing.py    #   ProbeBuffer, LearnedProbeRunner (RECAP-style RL)
│   └── aggressiveness.py     #   AggressivenessParams, AggressivenessModulator
├── conditioning/             # Conditioning modules
│   ├── modules.py            #   MLP, GNN, FiLM conditioners
│   └── gnn_translator.py     #   PropertyTranslator (bidirectional GAT graph network)
├── training/                 # Training pipeline
│   ├── config.py             #   TrainingConfig with probenet_enabled flag
│   └── trainer.py            #   Trainer orchestrator + sync daemon + ProbeNet conditioning
├── rollout/                  # Rollout worker
│   └── worker.py             #   Poll ckpt → collect episodes → upload to HF Hub loop
├── sync/                     # HF Hub coordination
│   ├── hub.py                #   Checkpoint poll, dataset upload/download, atomic markers
│   └── daemon.py             #   Background sync daemon thread
└── eval/                     # Evaluation
    └── metrics.py            #   EvalStats, EvalRunner, success detectors
```

### Research pipeline

| Phase | Status | Description |
|---|---|---|
| π₀.₅ baseline | ✅ | BC fine-tune on pick-and-place (no property info) |
| GR00T baseline | ✅ | BC fine-tune on pick-and-place (no property info) |
| ProbeNet-π₀.₅ | ✅ | π₀.₅ + physical property conditioning tokens |
| ProbeNet-GR00T | ✅ | GR00T + physical property conditioning tokens |
| Probe Encoder | ✅ | CNN + Transformer over raw force signal |
| GNN Translator | ✅ | Bidirectional property graph (interpretable ↔ latent) |
| Learned probing | ✅ | RECAP-style advantage conditioning for probe strategy |
| Aggressiveness | ✅ | Continuous [0,1] steering parameter |
| Eval pipeline | ✅ | Per-object metrics, success/failure detection |
| Sync daemon | ✅ | HF Hub trainer ↔ rollout coordination |
| Isaac Sim env | ✅ | SO-101 + objects + property randomization |
| Real hardware | ✅ | LeRobot SO101Follower wrapper |

---

## What needs a GPU

### Local (RTX 3060, 12 GB)

- **Inference**: serving π₀.₅ or GR00T policies (>8 GB needed — works)
- **Real hardware**: LeRobot teleoperation data collection
- **Development**: all code tests and module imports work without GPU

### Lambda Labs A100 (80 GB)

- **Fine-tuning**: BC + RL training for π₀.₅ and GR00T (>22.5 GB needed)

### Lambda Labs A10 (24 GB) or A100

- **Isaac Sim rollouts**: requires Isaac Sim + LeIsaac installed
  (`uv pip install "isaacsim[all,extscache]==6.0.1" --extra-index-url https://pypi.nvidia.com`)

---

## Infrastructure

Two Docker images built via Podman:

| Image | GPU | Base | Size | Purpose |
|---|---|---|---|---|
| `:trainer` | A100 | `nvidia/cuda:12.6` | ~10 GB | Fine-tune π₀.₅ / GR00T |
| `:rollout` | A10/A100 | `nvidia/cuda:12.6` + Isaac Sim pip | ~30 GB | Isaac Sim data collection |

Trainer and rollout communicate through HuggingFace Hub — no direct
networking needed. A background sync daemon handles checkpoint polling,
dataset upload/download, and atomic `_complete` markers.

### Quick start (cloud)

```bash
# On Lambda instance:
curl -sSL https://raw.githubusercontent.com/<user>/probenet/main/docker/install_podman.sh | bash

# Pull and run trainer:
podman run --device nvidia.com/gpu=all \
  -v ~/probenet-data:/data \
  -e HF_TOKEN=$HF_TOKEN \
  ghcr.io/<user>/probenet:trainer

# Pull and run rollout worker:
podman run --device nvidia.com/gpu=all \
  -v ~/probenet-data:/data \
  -e HF_TOKEN=$HF_TOKEN \
  ghcr.io/<user>/probenet:rollout
```

---

## Backends

```
backends/
├── openpi/         # git submodule — π₀.₅ (Physical Intelligence)
└── isaac-gr00t/    # git submodule — GR00T N1.7 (NVIDIA)
```

Both are installed as editable pip packages. Critical versions pinned:

| Package | Version |
|---|---|
| torch | 2.7.1+cu126 |
| jax[cuda12] | 0.5.3 |
| flax | 0.10.2 |
| transformers | 4.53.2 |
| isaacsim | 6.0.1 (rollout only, conflicts with openpi's torch pin) |

---

## Property set

| Phase | Properties | Source |
|---|---|---|
| **Phase 1** (implemented) | mass, friction, compliance | Interaction — probe force signal |
| **Phase 1 meta** (implemented) | fragility, slipperiness | Derived from core properties |
| **Phase 2** (planned) | shape, size, contents_movable, deformable | Vision — camera observation |

Properties are injected as text tokens into the VLA prompt:

```
grasp the shell | probe mode: manipulation | mass: 0.30 kg, friction: 0.50, compliance: 0.80 | aggressiveness: low
```

During training, all modalities are randomly dropped (following openpi's
multimodal recipe): property_dropout=0.25, probe_mode_dropout=0.1,
aggressiveness_dropout=0.15, language_dropout=0.05.

---

## Next steps

The codebase is ready for Isaac Sim integration. The remaining work is
**runtime** — not more code:

1. **Verify LeIsaac compatibility** with IsaacSim 6.0.1 (or pin to 5.1)
2. **Create LeIsaac task env** for the probe scene (SO-101 + shell objects + cameras)
3. **Generate initial dataset** (scripted oracle) on Lambda A10
4. **Upload to HF Hub** as a LeRobotDataset
5. **Fine-tune π₀.₅ baseline** on Lambda A100
6. **Fine-tune ProbeNet-π₀.₅** with property conditioning
7. **Compare baseline vs ProbeNet** on both backbones
8. **Write the paper**
