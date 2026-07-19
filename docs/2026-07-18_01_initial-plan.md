# Initial Planning Document for ProbeNet

## Purpose

This document is the first planning artifact for the ProbeNet project. It lays
out an initial execution approach for building the framework, simulation
environment, training pipeline, and real-hardware integration. This plan is
itself a draft — a "plan model" that can be discarded or rewritten as the
project evolves.

**Important:** This is an **initial planning model**, not the final
architecture. It should be treated as a starting scaffold that can be
rewritten, replaced, or discarded as the project evolves. Any component
here—file layout, module boundaries, sim parameters, or policy choices—can be
dropped in favor of a more suitable model once the evidence from experiments is
available.

## Scope

1. MuJoCo simulation using the `robotstudio_so101` model from
   [mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/robotstudio_so101).
2. Physical SO-101 hardware integration for real demonstrations.
3. A modular framework with swappable components:
   - Probe estimators (statistical baseline, 1D CNN).
   - Conditioning modules (MLP, GNN, FiLM).
   - Policy backbones (BC, ACT, Diffusion Policy).
4. A LeRobot-compatible dataset and training interface.
5. Simulation-first evaluation, followed by real-hardware transfer.

## Initial Execution Model

```
probenet/
├── env/          # MuJoCo scene loader, SO-101 arm, object set
├── probe/        # Scripted probe, signal logging, physical estimator
├── vision/       # Visual encoder and visual parameter extraction
├── conditioning/ # MLP / GNN / FiLM fusion modules
├── policies/     # LeRobot-compatible BC / ACT / Diffusion backbones
├── dataset/      # LeRobot dataset with custom probe/physics features
├── scripts/      # train.py, eval.py, collect_sim.py, collect_real.py
└── configs/      # Hydra / YAML experiment configs
```

### Environment layer (`probenet/env/`)

Load the `robotstudio_so101` Menagerie model, add a table, and place two
visually ambiguous shell objects (Shell A: fragile/light, Shell B: heavy/slippery).
Provide a simple Python API for reset, step, and rendering.

### Probe layer (`probenet/probe/`)

Implement a scripted probe:

1. Approach the object.
2. Squeeze with a controlled grip-force ramp.
3. Lift and hold for a fixed duration.
4. Release.

Log the probe signal (MuJoCo actuator forces / torques as the analog of servo
current) and estimate physical parameters: mass, compliance, and friction. The
estimator starts as a small MLP over hand-designed statistics (mean, slope, peak)
and can later be upgraded to a 1D CNN or Transformer.

### Vision layer (`probenet/vision/`)

Use a pretrained image encoder (ResNet, DINO, or ViT) with a small head to
produce visual parameters (shape, size, gloss, material class). In simulation,
visual parameters can be treated as known or pseudo-labeled; on real hardware,
they come from the vision encoder.

### Conditioning layer (`probenet/conditioning/`)

Fuse visual and physical parameters into a conditioning embedding:

- **MLP**: concatenate parameters and project.
- **GNN**: treat each parameter as a node and run message passing.
- **FiLM**: modulate visual features with physical parameters.

### Policy layer (`probenet/policies/`)

LeRobot-compatible policy backbones. Each backbone can be trained in two modes:

- **Baseline**: vision + proprioception → actions.
- **ProbeNet**: vision + proprioception + conditioning embedding → actions.

Backbones to support: BC, ACT, Diffusion Policy.

### Dataset layer (`probenet/dataset/`)

Extend the LeRobot dataset format with custom features:

- `visual_params`: constant per episode.
- `physical_params`: constant per episode (ground truth in sim, estimated in real).
- `probe_signal`: 1D time series stored as episode metadata.

### Scripts

- `scripts/collect_sim.py`: generate sim demonstrations with probe metadata.
- `scripts/collect_real.py`: collect real demonstrations with the SO-101 arm.
- `scripts/train.py`: train a policy or probe estimator.
- `scripts/eval.py`: evaluate a policy and produce success/failure metrics.

## Phases

1. **Sim setup and probe validation**: load SO-101, add objects, run probe,
   inspect signal differences between Shell A and Shell B.
2. **Baseline policy training**: train BC/ACT in simulation with vision + proprio.
3. **ProbeNet conditioning**: add MLP conditioning, then GNN, then FiLM.
4. **Simulation evaluation**: compare baseline vs. ProbeNet across shells and
   backbones.
5. **Real hardware transfer**: collect real data, run the probe with physical
   servos, fine-tune or evaluate the sim-trained models.

## Disclaimers

- This is an **initial plan model**, not a committed final design.
- Any component can be discarded or replaced if a more suitable approach is
  found during implementation.
- Simulation results are a stepping stone; real-hardware results are the target.
- The modular structure is chosen to enable experimentation, not to lock in a
  final architecture.

## Open Questions

- Which policy backbone should be implemented first? (Suggested: BC, then ACT.)
- Should Hydra be used for config management? (Suggested: yes, matching the
  existing `configs/` and `outputs/` directories.)
- Should the probe estimator be trained supervised on sim ground truth, or learned
  implicitly through the policy loss?
