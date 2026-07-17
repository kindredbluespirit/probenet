---
title: "ProbeNet"
---

# ProbeNet — Vision-Informed Interactive Perception

ProbeNet conditions a manipulation policy on object physical-parameter state (mass, friction, compliance) to generalize across a spectrum of objects with similar appearance but different physics.

## Pipeline

1. **Visual Perception** — Vision encoder observes the object → visual params (shape, size, gloss, coarse material class).

2. **Vision-Informed Interactive Perception (Probe Phase)** — Visual params set the probe's initial strategy. A scripted probe (squeeze + lift-hold) executes, reading servo current/torque → physical params (mass, compliance, true friction). This step can *confirm or correct* the visual prior.

3. **Manipulation Policy** — Conditioned on (visual params + physical params) via a parameter graph / GNN, feeding a BC or ACT action head. Executes pick-and-place with physics-aware trajectory and force.

## Headline Experiment

Two visually near-identical objects (e.g. matte fragile shell vs. weighted/slippery shell) with opposite correct grip-force/trajectory requirements. Baseline (vision-only) fails asymmetrically — crushes one, drops the other. ProbeNet succeeds on both by resolving the ambiguity through probing.

## Hardware

- SO-101 leader-follower arm pair
- Static-mounted RealSense D435if (third-person view, RGB stream)
- Servo current/torque readout used as the probe signal source

## Object Set

- **Shell A (fragile):** thin-wall PLA print, hollow, low mass
- **Shell B (rigid/heavy/slippery):** thick-wall or weighted-core print, smoother finish
- Same external geometry (~5-6cm sphere), painted to match
