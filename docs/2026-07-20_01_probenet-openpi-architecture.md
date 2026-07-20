# ProbeNet: Two-Stage Interactive Physical Perception with π₀.₇

## Architecture & Training Plan (July 20, 2026)

---

## 1. Motivation & Contribution

### Problem

VLA foundation models (π₀ through π₀.₇, OpenVLA, RT-2) map pixels + language → actions. They
learn *how* to do a task from demonstrations but have no mechanism to answer *what is this object
made of?* Two visually identical objects with different physical properties (mass, friction,
compliance, fragility) are indistinguishable — the policy cannot adapt its behavior.

Interactive perception (robots probing objects to extract physical properties) is well-studied
in classical robotics (Kruzliak et al. 2024) but has **never been integrated with a VLA
foundation model.**

### Contribution

ProbeNet extends π₀.₇ with a **two-stage interactive perception pipeline**:

1. **Stage 1 — Probing**: The model autonomously interacts with an object to estimate physical
   properties. Some properties come from vision alone (appearance changes like water spilling),
   others require interaction (mass, friction, fragility).
2. **Stage 2 — Manipulation**: The same model uses those property estimates as a new multimodal
   conditioning modality (like π₀.₇'s metadata tokens) to adapt its behavior.

The core architectural innovations:

- **Single shared VLA for both stages** — the same π₀.₇ backbone generates probing actions AND
  manipulation actions, distinguished by a probe-mode conditioning token
- **Physical property parameters as a first-class prompt modality** — extending π₀.₇'s
  multimodal framework (language + metadata + subgoal images) with object physics
- **Bidirectional Property Graph (GNN)** — a learnable translator between interpretable
  physical parameters and dense latent codes, enabling both sides of the system to work in
  their preferred representation
- **Learned probing policy** — not a scripted squeeze-lift sequence; the model chooses
  actions to maximize information gain
- **Aggressiveness steering** — a continuous [0,1] user parameter modulating risk tolerance in
  both probing and manipulation

---

## 2. Architecture

### 2.1 Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         π₀.₇ BACKBONE (shared)                            │
│  Gemma 3 VLM (~4B) + Action Expert (flow matching, ~860M)                 │
│                                                                            │
│  Prompt modalities (all randomly dropped during training):                 │
│  ┌────────────────────────────────────────────────────────────────────────┐│
│  │ 1. Images         — RGB from mounted camera(s)                         ││
│  │ 2. Language       — task instruction / subtask coaching                ││
│  │ 3. State          — proprioception (joint angles, velocities)          ││
│  │ 4. Visual subgoal — (π₀.₇ native: what the end state should look like)││
│  │ 5. Metadata       — (π₀.₇ native: speed, quality scores)              ││
│  │ 6. ★ PROBE MODE   — "probing" | "manipulation"                        ││
│  │ 7. ★ PHYSICAL     — {mass, friction, compliance, fragility, ...}       ││
│  │    PROPERTIES      │ + confidence per parameter                        ││
│  │ 8. ★ AGGRESSIVENESS— [0.0 ... 1.0]                                    ││
│  └────────────────────────────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────────────────────┘
         ▲                                              │
         │                                              ▼
   ┌─────┴──────────────────┐                    ┌──────────────┐
   │  PROPERTY TRANSLATOR   │                    │  ACTIONS     │
   │  (Bidirectional GNN)   │                    │  ┌─────────┐ │
   │                        │                    │  │ probing │  │
   │  interpretable ←→ latent│                   │  │  or     │  │
   │  params          code  │                    │  │ manip.  │  │
   └────────┬───────────────┘                    │  └─────────┘ │
            ▲                                    └──────┬───────┘
   ┌────────┴───────────────┐                          │
   │  PROBE ENCODER         │              ┌───────────▼───────────┐
   │  (1D CNN + Transformer)│              │  ENVIRONMENT           │
   │                        │◄─────────────┤  (MuJoCo sim / real)   │
   │  Inputs:               │              │                        │
   │  · Raw probe signal    │              │  Returns per step:     │
   │    (T, 6) actuator     │              │  · RGB image           │
   │    force over time     │              │  · Proprioceptive state│
   │  · Visual features     │              │  · Actuator force      │
   │    during probing      │              │    signal (from MuJoCo │
   │                        │              │    force sensors or    │
   │  Outputs:              │              │    real servo current) │
   │  · Physical params     │              └───────────────────────┘
   │  · Confidence per param│
   └────────────────────────┘
```

### 2.2 Component Details

#### A. π₀.₇ Backbone (shared, pre-trained)

- **VLM**: Gemma 3 (~4B params) — processes images, language, state. Frozen during
  initial phases, LoRA fine-tuned in later phases.
- **Action Expert**: Flow-matching transformer (~860M params) — denoises action chunks.
  Uses π₀.₇'s native prefix-suffix attention structure.
- **Prompt conditioning**: Following π₀.₇'s recipe, all prompt modalities are
  randomly dropped during training. At inference, any subset works.
- **New prompt modalities** (our additions):
  - `probe_mode ∈ {probing, manipulation}` — binary token appended to language prompt
  - `physical_props` — text tokens: `"mass: 0.3 kg, friction: 0.5, compliance: 0.8,
    fragility: 0.2, slipperiness: 0.7"` — tokenized into π₀.₇'s language embedding
    space and appended to prefix
  - `aggressiveness ∈ [0, 1]` — tokenized as text: `"aggressiveness: high"` or
    `"aggressiveness: 0.3"`, appended to prefix

#### B. Probe Encoder

A standalone module that processes interaction history to estimate physical properties.

**Inputs**:
- **Raw probe signal**: `(T, signal_dim)` tensor — actuator force / servo current over
  time. `signal_dim = nu` (6 for SO-101: shoulder_pan/lift, elbow/wrist_flex/roll,
  gripper). Also `qfrc_actuator` from MuJoCo.
- **Visual features during probing**: Initial image embedding + change features
  (e.g., difference between frame at t=0 and t=T, for detecting motion, deformation,
  spilled contents). Extracted from π₀.₇'s frozen visual encoder.
- **Proprioceptive history**: Joint states during probing, to contextualize forces.

**Architecture**:
```
┌──────────────────────────────────────────────────┐
│  Raw signal (T, 6)       Visual features          │
│  ┌─────────────────┐    ┌─────────────────┐      │
│  │ 1D CNN          │    │ π₀.₇ visual     │      │
│  │ (kernel=7,       │    │ encoder (frozen) │      │
│  │  stride=2,       │    │ → image_embed    │      │
│  │  channels=32→64) │    │ → delta_embed    │      │
│  │ → (T', 64)       │    └────────┬────────┘      │
│  │                  │             │                │
│  │ Transformer      │             │                │
│  │ (2 layers,       │             │                │
│  │  4 heads)        │◄────────────┘                │
│  │ → (1, 128)       │  cross-attention to visual   │
│  └────────┬─────────┘                              │
│           │                                        │
│  ┌────────▼────────────────────────────────────┐  │
│  │  Prediction Heads                            │  │
│  │  ┌──────────────┐ ┌──────────────┐          │  │
│  │  │ Param Head   │ │ Conf Head    │          │  │
│  │  │ Linear→mass  │ │ Linear→σ     │          │  │
│  │  │ Linear→fric  │ │   per param  │          │  │
│  │  │ Linear→comp  │ └──────────────┘          │  │
│  │  │ Linear→frag  │                            │  │
│  │  │ Linear→slip  │                            │  │
│  │  │ ... (extensible)│                          │  │
│  │  └──────────────┘                            │  │
│  └──────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────┘
```

**Output**: `{param_name: (value, confidence)}` dictionary, plus a concatenated dense
vector `(num_params, 2)` = `(value, log_confidence)` for the GNN translator.

#### C. Property Translator (Bidirectional GNN)

The bridge between interpretable physical parameters and the dense latent code that
π₀.₇'s action expert operates on.

```
   INTERPRETABLE SPACE                   LATENT SPACE

   ┌─────────┐ ┌───────────┐ ┌──────────┐ ┌──────────┐
   │  mass   │ │ friction  │ │complianc.│ │fragility │  ...
   │ node₀   │ │  node₁    │ │  node₂   │ │  node₃   │
   │ (32-dim)│ │ (32-dim)  │ │ (32-dim) │ │ (32-dim) │
   └────┬────┘ └─────┬─────┘ └────┬─────┘ └────┬─────┘
        │            │            │            │
        └────────────┼────────────┼────────────┘
                     │            │
              ┌──────┴────────────┴──────┐
              │   Message Passing         │
              │   (2 rounds, GAT-style)   │
              │                           │
              │   + Global LATENT NODE    │
              │   (128-dim)               │
              └──────────┬───────────────┘
                         │
                  ┌──────┴──────┐
                  │  LATENT CODE │  → injected into π₀.₇ property tokens
                  │  (128-dim)   │
                  └─────────────┘

   Reverse path: latent code → message passing → per-param nodes → value heads
```

**Bidirectional operation**:
- **Forward** (probe → properties → latent): Probe Encoder outputs per-param values
  and confidences → each becomes a node embedding → message passing → global latent
  node → injected into π₀.₇ prompt.
- **Reverse** (latent → properties): Global latent node → message passing to per-param
  nodes → value prediction heads → interpretable parameters (for inspection,
  debugging, and training regularization).
- **Straight-through** (probe → latent): Probe Encoder → directly predict latent code,
  bypassing the GNN's intermediate parameterization. Faster, less interpretable.
  Used as an alternative path during joint training.

#### D. Probing Policy (within π₀.₇)

The probing stage is not a separate model. π₀.₇ generates probing actions when
conditioned with `probe_mode = "probing"`:

```
Probing prompt:
  Language: "Explore this object to understand its physical properties."
  Probe mode: "probing"
  Aggressiveness: 0.3
  Physical properties: <empty / zeros>  (nothing known yet)
  Visual subgoal: <none>
```

The model generates a sequence of actions over a fixed probing horizon `T_probe`
(e.g., 100–300 steps). After probing completes, the Probe Encoder processes the
recorded signal + visual history and produces property estimates.

**Key**: the probing actions are *learned*, not scripted. The training signal comes
from:
1. **Property estimation accuracy** (supervised — ground truth params are known in sim)
2. **Information gain** (RL-style — value function estimates how much a probing action
   reduces parameter uncertainty)
3. **Downstream task success** (reinforcement — better probing → better manipulation)

#### E. Manipulation Policy (within π₀.₇)

After probing, the estimated properties are injected back into π₀.₇'s prompt:

```
Manipulation prompt:
  Language: "Pick up the object and place it on the shelf."
  Probe mode: "manipulation"
  Aggressiveness: 0.3
  Physical properties: "mass: 0.30, friction: 0.50, compliance: 0.80,
                        fragility: 0.20, slipperiness: 0.70"
  Visual subgoal: <end-state image>
```

The model adapts grip force, approach speed, trajectory smoothness, and release
strategy based on the estimated parameters.

---

## 3. Property Set

### 3.1 Core Properties (Phase 1 — simulation)

| Property | Range | Source | Description |
|---|---|---|---|
| `mass` | 0.01–5.0 kg | Interaction (lift force) | How heavy is it? |
| `friction` | 0.0–2.0 | Interaction (slide resistance) | Is it slippery or grippy? |
| `compliance` | 0.0–1.0 | Interaction (deformation under force) | Is it soft or rigid? (maps to MuJoCo solref/solimp) |
| `fragility` | 0.0–1.0 | Meta-property (derived from compliance + mass) | How likely to break under force? |
| `slipperiness` | 0.0–1.0 | Derived from friction + surface | How easily does it slip from gripper? |

### 3.2 Vision-derived Properties (Phase 2 — when visual encoder is added)

| Property | Range | Source | Description |
|---|---|---|---|
| `shape` | categorical + params | Vision | Box, sphere, cylinder, irregular |
| `size` | 0.01–0.5 m | Vision | Dimensions |
| `appearance` | categorical | Vision | Texture, color, reflectivity |
| `contents_movable` | 0.0–1.0 | Vision (change detection) | Does the object contain liquid/granular material that moves? |
| `deformable` | 0.0–1.0 | Vision + interaction | Does it change shape? |

### 3.3 Extensible design

The property set is not fixed. New properties can be added by:
1. Adding a node to the GNN
2. Adding a prediction head to the Probe Encoder
3. Adding a new token to the physical property prompt
4. Training on data that includes that property

Parameters that can't be estimated (not relevant to the object, or probing insufficient)
are marked with low confidence and masked/excluded from the prompt.

---

## 4. Training Pipeline

### 4.1 Training Data

#### Simulation Data (Phase 1)

Generated in MuJoCo with the SO-101 arm. Each episode consists of a probing phase
followed by a task phase:

```
Episode structure:
  ┌─────────────────────────────────────────────────────────┐
  │ Resample object properties from distribution              │
  │   mass ~ U(0.01, 5.0), friction ~ U(0.0, 2.0),          │
  │   compliance ~ U(0.0, 1.0)                               │
  ├─────────────────────────────────────────────────────────┤
  │ Phase 1: PROBING (T_probe steps, scripted oracle)        │
  │   - Oracle probing actions (tap, squeeze, lift, slide)   │
  │   - Record: images, state, actions, actuator forces      │
  │   - Ground truth: known physical params                  │
  │   - Aggressiveness: fixed per episode                    │
  ├─────────────────────────────────────────────────────────┤
  │ Phase 2: TASK (T_task steps, scripted oracle)            │
  │   - Pick-and-place or other task                         │
  │   - Oracle adapts behavior based on ground truth params  │
  │   - Record: images, state, actions                       │
  │   - Physical params: ground truth (for supervision)      │
  │   - Aggressiveness: same as probing                      │
  └─────────────────────────────────────────────────────────┘
```

**Data format**: LeRobot format (HuggingFace datasets), following openpi conventions.

**Dataset size target**: 5,000–10,000 episodes with diverse property combinations.
Each episode ~1,000 steps (300 probing + 700 task).

#### Learning-to-Probe Data (Phase 2)

Once the Probe Encoder is reliable, we replace the scripted oracle probing with
autonomous policy rollouts:

- Deploy π₀.₇ in probing mode
- Collect probing trajectories (images + signals + actions)
- Human/sim labels: episode is "successful probe" if downstream task succeeds
- Feed back into training as RECAP-style advantage data

### 4.2 Training Phases

```
PHASE 0: PROBE ENCODER PRE-TRAINING (supervised)
══════════════════════════════════════════════════
Train Probe Encoder to predict physical params from scripted probe signals.
No VLA involved — pure supervised regression.

Loss: L_MSE(params_pred, params_gt) + L_conf(confidence)
Data: Scripted oracle probing episodes
Output: Pre-trained Probe Encoder checkpoint

                              │
                              ▼

PHASE 1: PROPERTY GRAPH PRE-TRAINING (self-supervised + reconstruction)
════════════════════════════════════════════════════════════════
Train the GNN translator for bidirectional latent ↔ interpretable mapping.

Loss: L_recon(params → latent → reconstructed_params)
    + L_cycle(latent → params → reconstructed_latent)
    + L_contrast (similar params → similar latent, different params → far apart)

                              │
                              ▼

PHASE 2: π₀.₇ MULTIMODAL FINE-TUNING
══════════════════════════════════════════════════
Fine-tune π₀.₇ with physical properties as a new prompt modality.
Following π₀.₇'s multimodal recipe: random dropout of ALL modalities.

Crucially, at this stage the Probe Encoder and GNN are FROZEN.
Only π₀.₇ weights are updated (LoRA on the VLM backbone).

Loss: L_flow (π₀.₇ native flow-matching loss on action prediction)
Modality dropout schedule:
  - probe_mode: 10% dropout
  - physical_props: 25% dropout (tied — if dropped, probe_mode also dropped)
  - aggressiveness: 15% dropout
  - language: 5% dropout (small — task identity is usually needed)
  - images: 0% dropout (always present)

At test time: can run with any subset of modalities.

                              │
                              ▼

PHASE 3: LEARNED PROBING POLICY (RECAP-style RL)
══════════════════════════════════════════════════
Replace scripted oracle probing with learned probing.
Train the probing policy using RECAP's advantage conditioning.

For this phase only, we add a "probe_advantage" token:
  - positive probe: downstream task succeeded
  - negative probe: downstream task failed

This is structurally parallel to RECAP:
  - RECAP: advantage predicts task success → conditions action quality
  - ProbeNet: probe_advantage predicts task success after probing → conditions
    probing strategy

1. Deploy π₀.₇ in probing mode → collect probing trajectories
2. Run downstream task with estimated properties
3. Label probing trajectories as "positive probe" or "negative probe" based on
   downstream task outcome
4. Fine-tune π₀.₇ on labeled probing data (LoRA)
5. Repeat

                              │
                              ▼

PHASE 4: JOINT END-TO-END FINE-TUNING (optional)
══════════════════════════════════════════════════
Unfreeze Probe Encoder + GNN + π₀.₇ (LoRA).
Jointly optimize all components.

Loss: L_flow + α * L_param_estimation + β * L_latent_consistency

This phase is optional and depends on whether we have enough data for
stable joint optimization.

                              │
                              ▼

PHASE 5: REAL-WORLD TRANSFER (future, not in initial scope)
════════════════════════════════════════════════════════════
- Fine-tune Probe Encoder on real servo current data
- Collect real-world probing + task demonstrations
- Domain-adapt property distributions (sim → real gap)
```

### 4.3 Loss Functions Summary

| Phase | Loss | Description |
|---|---|---|
| 0 | `MSE(params_pred, params_gt) + λ * NLL(confidence)` | Supervised param estimation with uncertainty |
| 1 | `MSE(cycle_params) + MSE(cycle_latent) + NT-Xent(contrast)` | Graph bi-directional consistency + contrastive |
| 2 | `Flow_matching_loss(actions_pred, actions_gt)` | π₀.₇ native loss — no additional terms |
| 3 | `Flow_matching_loss + λ * MSE(advantage_pred, advantage_gt)` | RECAP-style advantage conditioning |
| 4 | `L_flow + α * L_param + β * L_graph_consistency` | Joint optimization |

---

## 5. Inference Pipeline

### 5.1 Stage 1 — Probing

```python
def probe_object(observation, aggressiveness=0.3, max_probe_steps=300):
    """
    Run autonomous probing and return physical property estimates.
    """
    history = {"images": [], "states": [], "actions": [], "signals": []}

    for t in range(max_probe_steps):
        # Build prompt with no property info (unknown object)
        prompt = build_prompt(
            images=observation["image"],
            state=observation["state"],
            language="Explore this object to understand its physical properties.",
            probe_mode="probing",
            physical_props=None,  # ← nothing known yet
            aggressiveness=aggressiveness,
        )

        action = pi_model(prompt)
        observation, signal = env.step(action)

        history["images"].append(observation["image"])
        history["states"].append(observation["state"])
        history["actions"].append(action)
        history["signals"].append(signal)

        # Terminate early if confidence exceeds threshold?
        if t >= 50:  # minimum probing
            params, confidence = probe_encoder(history)
            if all(c > 0.9 for c in confidence.values()):
                break

    params, confidence = probe_encoder(history)
    latent = property_translator.forward(params, confidence)
    return params, confidence, latent
```

### 5.2 Stage 2 — Manipulation

```python
def manipulate_object(observation, params, confidence, aggressiveness=0.3,
                      task="Pick up and place on shelf"):
    """
    Execute task with physical property conditioning.
    """
    # Filter low-confidence params (don't condition on what we don't know)
    known_params = {k: v for k, v in params.items()
                    if confidence[k] > 0.5}

    # Run π₀.₇ with property-aware prompt
    while not done:
        prompt = build_prompt(
            images=observation["image"],
            state=observation["state"],
            language=task,
            probe_mode="manipulation",
            physical_props=known_params,
            aggressiveness=aggressiveness,
        )

        action = pi_model(prompt)
        observation, reward, done, info = env.step(action)
```

---

## 6. Impact of Aggressiveness

The aggressiveness parameter `agg ∈ [0, 1]` modulates both stages:

### During probing

| agg | Behavior |
|---|---|
| 0.0 | Gentle taps, slow squeezes, minimal force. Stop early if uncertain. |
| 0.3 | Moderate pressure, standard sequence. |
| 0.7 | Firm squeezes, shake test, lift-and-drop. Higher force tolerance. |
| 1.0 | Destructive testing allowed. Squeeze until deformation/breakage. Drop test. |

In sim, `agg = 1.0` means actions that would break the object are acceptable during
probing. The model learns that for `agg = 0.0`, probe actions must keep forces below
safe thresholds.

### During manipulation

| agg | Behavior |
|---|---|
| 0.0 | Gentle grip, slow movements, wide safety margins. Lower throughput. |
| 1.0 | Fast movements, tight grip, narrow margins. Higher throughput, higher risk. |

This is implemented as a conditioning token in π₀.₇'s prompt (similar to how π₀.₇
already uses "speed" and "quality" metadata). During training, aggressiveness is
randomized per episode, and demonstration data for each level is collected.

---

## 7. Property Visualization & Interpretability

The bidirectional GNN enables:

- **At inference**: the latent code flowing through π₀.₇ can be decoded back to
  interpretable physical parameters for visualization
- **For debugging**: if the policy behaves unexpectedly, check what physical
  parameters it "thinks" the object has
- **For safety**: monitor parameter estimates and confidence; refuse to manipulate
  if critical parameters are uncertain
- **For the paper**: show the correlation between estimated parameters and
  ground-truth parameters across objects and probing strategies

---

## 8. Implementation Plan

### Step 1: Data Generation Infrastructure
- Extend `collect_sim.py` to produce two-phase episodes (probing + task)
- Add property randomization across a continuous distribution
- Add aggressiveness annotation per episode
- Add LeRobot format export
- Target: 5,000 episodes with diverse properties

### Step 2: Probe Encoder
- Implement `ProbeEncoder` class in `src/probenet/probe/`
- CNN + Transformer over raw signal with visual cross-attention
- Supervised training on scripted oracle data
- Validate: can predict mass/friction/compliance from probe signal alone?

### Step 3: Property Translator (GNN)
- Extend existing `conditioning/modules.py` `GNNConditioner`
- Add bidirectional capability (interpretable → latent and latent → interpretable)
- Add reconstruction + contrastive training
- Validate: can the graph reconstruct parameters from arbitrary latent codes?

### Step 4: π₀.₇ Integration
- Add new prompt modalities (`probe_mode`, `physical_props`, `aggressiveness`)
- Implement modality dropout in the data pipeline
- Convert ProbeNet dataset to LeRobot format with property annotations
- Fine-tune π₀.₇ (LoRA) with frozen Probe Encoder + GNN
- Validate: does property conditioning change manipulation behavior appropriately?

### Step 5: Learned Probing
- Replace scripted oracle with autonomous probing rollouts
- Implement RECAP-style advantage conditioning for probing quality
- Iterative improvement loop (collect → label → retrain)
- Validate: does learned probing outperform scripted oracle?

### Step 6: Ablation Studies
- No properties (baseline): π₀.₇ with no physical property conditioning
- Scripted probe: Scripted squeeze-lift + Probe Encoder → conditioned π₀.₇
- Learned probe (ours): Learned probing → Probe Encoder → conditioned π₀.₇
- Aggressiveness sweep: 0.0, 0.3, 0.5, 0.7, 1.0
- Interpretability: GNN latent → param reconstruction quality

### Step 7: Real Hardware (future)
- Port Probe Encoder to real servo current signals
- Collect real-world two-phase demos
- Sim-to-real transfer of property-conditioned policy

---

## 9. Open Questions

1. **How many properties for Phase 1?** Start with mass + friction + compliance (3 in
   current sim), add fragility and slipperiness as derived meta-properties. Expand
   to vision-derived properties (Phase 2).

2. **Probe horizon (T_probe)?** Can the policy terminate probing early when it's
   confident? Needs a learned termination condition.

3. **Task diversity in Phase 1?** Multiple tasks per object, or single pick-and-place?
   Single task is simpler for validation.

4. **Property distribution coverage?** Uniform sampling may miss edge cases (very
   heavy + very fragile, very light + very grippy). Need to ensure coverage.

5. **Aggressiveness data collection?** How do we get demonstrations for agg=1.0
   (destructive)? In sim, scripted. In real, much harder — may need sim-only or
   synthetic data.

6. **Confidence threshold for Stage 2?** If probing fails (low confidence on key
   parameters), should the policy refuse, retry probing, or fall back to a safe
   default?

---

## 10. Relationship to Existing Literature

| Paper | Our relationship |
|---|---|
| **π₀ (Black et al. 2024)** | We use π₀'s flow-matching VLA architecture as the backbone |
| **π₀.₇ (Physical Intelligence, April 2026)** | We extend π₀.₇'s multimodal prompt framework with physical property tokens; our probing stage is analogous to π₀.₇'s language coaching but for physical discovery |
| **RECAP / π\*₀.₆ (Nov 2025)** | We adopt RECAP's advantage-conditioning pattern for training the learned probing policy ("positive probe" / "negative probe" tokens) |
| **Knowledge Insulation (π₀.₅)** | The AdaRMS conditioning mechanism in π₀.₅ could serve as an alternative injection point for property embeddings alongside or instead of prompt tokens |
| **Kruzliak et al. (IROS 2024)** | Interactive perception without VLA integration — our work bridges this to foundation models |
| **Hi Robot (π, Feb 2025)** | Hierarchical VLA with high-level policy generating language subgoals — our Stage 1 (probing) → Stage 2 (manipulation) is a different kind of hierarchy (information gathering → execution) |
