"""Physical property definitions, ranges, and token builders.

Defines the canonical set of physical properties that ProbeNet estimates
during Stage 1 (probing) and uses to condition manipulation in Stage 2.

Each property has a range, a unit, a source (interaction vs vision-derived),
and a token format for injection into the VLA prompt.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Literal

PropertySource = Literal["interaction", "vision", "meta"]


@dataclass
class PropertyDef:
    """Definition of a single physical property."""

    key: str
    label: str  # human-readable name for prompt tokens
    low: float
    high: float
    unit: str
    source: PropertySource = "interaction"
    confidence_threshold: float = 0.5  # min confidence to include in prompt


# ── Core (interaction-derived) properties ─────────────────────────────────────

CORE_PROPERTIES: list[PropertyDef] = [
    PropertyDef("mass", "mass", 0.01, 5.0, "kg", "interaction", 0.3),
    PropertyDef("friction", "friction", 0.0, 2.0, "", "interaction", 0.5),
    PropertyDef("compliance", "compliance", 0.0, 1.0, "", "interaction", 0.5),
]

# ── Meta (derived) properties ─────────────────────────────────────────────────

META_PROPERTIES: list[PropertyDef] = [
    PropertyDef("fragility", "fragility", 0.0, 1.0, "", "meta", 0.5),
    PropertyDef("slipperiness", "slipperiness", 0.0, 1.0, "", "meta", 0.5),
]

# ── Vision-derived properties (Phase 2) ───────────────────────────────────────

VISION_PROPERTIES: list[PropertyDef] = [
    PropertyDef("shape", "shape", 0.0, 1.0, "", "vision", 0.5),
    PropertyDef("size", "size", 0.01, 0.5, "m", "vision", 0.5),
    PropertyDef("contents_movable", "contents_movable", 0.0, 1.0, "", "vision", 0.5),
    PropertyDef("deformable", "deformable", 0.0, 1.0, "", "vision", 0.5),
]

ALL_PROPERTIES: list[PropertyDef] = CORE_PROPERTIES + META_PROPERTIES + VISION_PROPERTIES
CORE_KEYS = {p.key for p in CORE_PROPERTIES}


@dataclass
class PropertyState:
    """Estimated or ground-truth physical properties for one object.

    Each value is paired with a confidence in [0, 1]. Low-confidence
    entries are excluded from the prompt.
    """

    values: dict[str, float] = field(default_factory=dict)
    confidences: dict[str, float] = field(default_factory=dict)

    def known_properties(self, threshold: float = 0.5) -> dict[str, float]:
        """Return properties with confidence above threshold."""
        return {k: self.values[k] for k in self.values if self.confidences.get(k, 0.0) >= threshold}

    @classmethod
    def from_ground_truth(cls, values: dict[str, float]) -> PropertyState:
        """Create a state with perfect confidence (sim ground truth)."""
        return cls(values=dict(values), confidences={k: 1.0 for k in values})

    @classmethod
    def empty(cls) -> PropertyState:
        """Create an empty state (nothing known)."""
        return cls(values={}, confidences={})


# ── Token builders ────────────────────────────────────────────────────────────


def properties_to_prompt(
    props: PropertyState,
    threshold: float = 0.5,
) -> str:
    """Convert physical properties to a text prompt fragment.

    Format: ``"mass: 0.30 kg, friction: 0.50, compliance: 0.80"``

    Args:
        props: Estimated or ground-truth property state.
        threshold: Minimum confidence to include a property.

    Returns:
        Comma-separated property string, or empty if nothing known.
    """
    known = props.known_properties(threshold)
    if not known:
        return ""

    parts: list[str] = []
    for pdef in ALL_PROPERTIES:
        if pdef.key in known:
            val = known[pdef.key]
            if pdef.unit:
                parts.append(f"{pdef.label}: {val:.2f} {pdef.unit}")
            else:
                parts.append(f"{pdef.label}: {val:.2f}")
    return ", ".join(parts)


def aggressiveness_to_token(value: float) -> str:
    """Convert continuous aggressiveness to a prompt token.

    Maps [0, 1] to a categorical label for the VLA prompt.
    """
    if value <= 0.2:
        return "aggressiveness: very low"
    elif value <= 0.4:
        return "aggressiveness: low"
    elif value <= 0.6:
        return "aggressiveness: medium"
    elif value <= 0.8:
        return "aggressiveness: high"
    else:
        return "aggressiveness: very high"


def probe_mode_token(mode: str) -> str:
    """Return the probe mode prompt token."""
    return f"probe mode: {mode}"


def build_probenet_prompt(
    task: str,
    properties: PropertyState,
    aggressiveness: float = 0.3,
    probe_mode: str = "manipulation",
) -> str:
    """Build the full ProbeNet prompt with all conditioning tokens.

    Args:
        task: Task description (e.g. "Pick up the object and place it").
        properties: Physical property estimates.
        aggressiveness: Risk tolerance [0, 1].
        probe_mode: ``"probing"`` or ``"manipulation"``.

    Returns:
        Full prompt string with all conditioning tokens appended.
    """
    parts = [task, probe_mode_token(probe_mode)]

    props_str = properties_to_prompt(properties)
    if props_str:
        parts.append(props_str)

    parts.append(aggressiveness_to_token(aggressiveness))
    return " | ".join(parts)


# ── Modality dropout (training) ───────────────────────────────────────────────


def drop_modalities(
    prompt: str,
    probe_mode_dropout: float = 0.1,
    property_dropout: float = 0.25,
    aggressiveness_dropout: float = 0.15,
    language_dropout: float = 0.05,
) -> str:
    """Randomly drop ProbeNet conditioning tokens from the prompt.

    Following openpi's multimodal recipe: all modalities are randomly dropped
    during training so the model learns to use any subset at inference.

    Args:
        prompt: The full prompt string built by ``build_probenet_prompt``.
        probe_mode_dropout: Probability of dropping the probe mode token.
        property_dropout: Probability of dropping physical property tokens.
            When dropped, probe_mode is also dropped (tied).
        aggressiveness_dropout: Probability of dropping aggressiveness token.
        language_dropout: Probability of dropping the task description.

    Returns:
        Prompt with some tokens removed.
    """
    parts = prompt.split(" | ")

    if random.random() < property_dropout:
        # Drop both properties AND probe_mode (tied)
        parts = [p for p in parts if "probe mode" not in p and "mass:" not in p and "friction:" not in p and "fragility:" not in p and "slipperiness:" not in p and "compliance:" not in p and "deformable:" not in p and "contents_movable:" not in p and "shape:" not in p and "size:" not in p]
    elif random.random() < probe_mode_dropout:
        parts = [p for p in parts if "probe mode" not in p]

    if random.random() < aggressiveness_dropout:
        parts = [p for p in parts if "aggressiveness" not in p]

    if random.random() < language_dropout:
        # Keep at least one token (drop the task description)
        parts = [p for p in parts if p == parts[0] is False or (p != parts[0] and len(parts) > 1)]
        # Simpler: just clear the first part (task description)
        if parts:
            parts[0] = ""

    result = " | ".join(p for p in parts if p)
    return result
