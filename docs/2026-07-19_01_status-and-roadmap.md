# ProbeNet: Status & Roadmap (July 19 2026)

## 1. Current State

### What exists

The project has a working pipeline from simulation data generation to policy
training. The core environment, scripted probe, dataset infrastructure, and two
policy variants (baseline BC and ProbeNet-BC with MLP conditioning) are all
implemented and verified.

### Key numbers

| Metric | Value |
|---|---|
| Sim dataset | 100 episodes (50 shell_a, 50 shell_b) |
| Shell A param | mass=0.05 kg, friction=1.2, compliant |
| Shell B param | mass=0.3 kg, friction=0.3, rigid |
| Probe contrast | 2517% mean actuator-force difference |
| Baseline BC val loss (30 epochs) | 0.000204 |
| ProbeNet val loss (30 epochs) | 0.000304 |
| GPU | RTX 3060 12 GB, CUDA 13.0 |
| Tests | 7/7 passing |
| Lint | ruff clean across all code |

## 2. Architecture Overview

```
probenet/
├── env/              # SO-101 MuJoCo env (gymnasium interface)
│   └── so101_env.py  #     two visually identical shell objects
├── probe/            # Scripted probe + signal feature extraction
│   └── probe_runner.py
├── vision/           # (placeholder — no implementation yet)
├── conditioning/     # MLP / GNN / FiLM modules
│   └── modules.py
├── policies/         # Baseline BC + ProbeNetPolicy wrapper
│   └── bc_policy.py
├── dataset/          # EpisodicDataLoader, serialization
│   └── sim_dataset.py
├── utils/            # Paths for third-party assets
│   └── paths.py
├── scripts/
│   ├── collect_sim.py    # Generate sim demos with probe
│   ├── train.py          # Train baseline / ProbeNet policies
│   ├── eval.py           # Evaluate a saved checkpoint
│   └── verify_probe.py   # Quick probe-signal test
├── docs/             # Planning artifacts
├── third_party/      # (gitignored) mujoco_menagerie clone
└── data/             # (gitignored) datasets
```

## 3. Working pieces vs. placeholders

| Component | Status | Notes |
|---|---|---|
| MuJoCo SO-101 env | **Done** | Two visually identical shell objects with different physics |
| Scripted probe | **Done** | Squeeze + lift-hold sequence, actuator-force logging |
| Probe feature extraction | **Done** | Mean/max/std of force signals |
| Visual encoder | **Placeholder** | Params hardcoded — no CNN processes the images |
| Conditioning: MLP | **Done** | Working and trained |
| Conditioning: GNN | **Written** | Module exists, not tested in training pipeline |
| Conditioning: FiLM | **Written** | Module exists, not tested |
| BC policy | **Done** | Simple CNN + MLP, trains to ~0.0002 MSE |
| ProbeNet policy wrapper | **Done** | Injects conditioning embedding before action head |
| Dataset collection | **Done** | Scripted oracle with joint-space interpolation |
| Training loop | **Done** | EpisodicDataLoader, MSE loss, checkpointing |
| Evaluation | **Basic** | Runs policy in env, but no reward/termination yet |
| Real hardware integration | **Not started** | |

## 4. Known Limitations

1. **Vision encoder is unimplemented.** Visual parameters are hardcoded and
   identical for both shells. The baseline cannot distinguish shells visually,
   which is correct for the experiment, but a real vision encoder would be
   needed for arbitrary objects.

2. **No success/failure detection.** The environment has no reward or
   termination condition, so the evaluation script reports max steps (500) for
   every trial, making it impossible to compare policy quality numerically.

3. **Scripted oracle is simplistic.** The pick-and-place trajectory uses fixed
   keyframes interpolated in joint space. It does not adapt to different object
   positions or arm configurations. This limits demo diversity.

4. **Data collection is slow.** Each episode re-creates the MuJoCo model from
   XML (~14 s/episode). Reusing the model instance across episodes would cut
   this by an order of magnitude.

5. **Action space mismatch.** The policy predicts normalized joint targets,
   while the env expects normalized actions in [-1, 1]. The mapping works but
   is fragile — the same pipeline with a different robot or action space would
   need changes.

6. **No validation of probe estimation.** The probe distinguishes shells but
   the extracted features are not validated against ground-truth mass/friction.
   There is no trained physical-parameter estimator yet — features are used
   directly.

## 5. Immediate Future Tasks (Suggested Ordering)

### Priority 1 — Fix evaluation

- Add a termination condition to the env (e.g., object within a target zone
  for success, object fallen for failure).
- Record failure mode (drop vs. not-reached vs. collision).
- Update `eval.py` to report per-shell success rates instead of step counts.

### Priority 2 — Implement vision encoder

- Add a pretrained backbone (ResNet-18, DINO, or small ViT) in
  `probenet/vision/`.
- Train a small head that predicts the 4 visual params (shape, size, gloss,
  material) from RGB images.
- Wire into the data pipeline so visual params come from actual vision.

### Priority 3 — Complete conditioning variants

- Run training + eval with GNN conditioner.
- Run training + eval with FiLM conditioner.
- Compare all three against baseline.

### Priority 4 — Improve data generation

- Replace hardcoded keyframes with an IK-based oracle that can reach any
  object pose.
- Cache the MuJoCo model instance between episodes to speed collection.
- Add randomization of the drop target location.

### Priority 5 — Train probe estimator

- Build a small MLP that maps probe features → (mass, compliance, friction).
- Train supervised on sim data where ground truth is known.
- Validate that estimated params match the ground-truth shell assignment.

### Priority 6 — Add ACT / Diffusion Policy backbones

- Add `ACT` and `DiffusionPolicy` classes to `probenet/policies/`.
- Train and evaluate alongside BC on the sim dataset.

### Priority 7 — Real hardware

- Port the probe script to the physical SO-101 arm (servo current as signal).
- Collect real demos via LeRobot teleop pipeline.
- Fine-tune or run sim-trained policies on real hardware.
- Compare sim vs. real probe signals.

## 6. Longer-Term Roadmap

1. Hybrid sim+real training (augment real data with sim data).
2. Full evaluation with publication-quality figures: failure tables, rollout
   clips, probe-signal waveforms for both shells.
3. Ablation study: which conditioning module works best and why.
4. Add more object types (more than 2 shells) to test generalization.
5. Project page update with real results and videos.
6. Paper draft positioning against interactive-perception literature.

## 7. Open Questions

- Should the env reward be sparse (1 for success, 0 otherwise) or dense (distance
  to target)?
- Which vision backbone fits the RTX 3060 budget for real-time inference:
  ResNet-18, a small ViT, or DINOv2?
- Should we write the real-hardware data collection script now, or iterate in
  simulation until the policy quality is satisfactory?
- For the probe estimator, is the statistical baseline (mean/max/std of forces)
  sufficient, or should we move directly to a 1D CNN on the raw signal?
