"""ProbeNet conditioning — physical property tokens for VLA policies.

Integrates with both π₀.₅ and GR00T backends by augmenting observation
dicts with property-conditioned prompts.  Used during training (with
dropout) and inference (with sensor data).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from probenet.probe.properties import (
    PropertyState,
    build_probenet_prompt,
    drop_modalities,
)


@dataclass
class ProbeNetConfig:
    """Configuration for ProbeNet conditioning.

    Controls dropout rates, property ranges, and aggressiveness defaults.
    """

    # Dropout rates (following openpi multimodal recipe)
    probe_mode_dropout: float = 0.1
    property_dropout: float = 0.25
    aggressiveness_dropout: float = 0.15
    language_dropout: float = 0.05

    # Defaults for inference
    default_aggressiveness: float = 0.3
    default_probe_mode: str = "manipulation"

    # Whether to use properties at all (baseline = False)
    enabled: bool = True


@dataclass
class ProbeNetConditioner:
    """Augments observation dicts with property-conditioned prompts.

    During training, properties are read from the dataset (ground truth).
    During inference, properties come from the probe encoder output.
    """

    config: ProbeNetConfig = field(default_factory=ProbeNetConfig)

    def augment_observation(
        self,
        obs: dict,
        properties: PropertyState | None = None,
        aggressiveness: float | None = None,
        probe_mode: str | None = None,
        training: bool = False,
    ) -> dict:
        """Add ProbeNet conditioning tokens to an observation dict.

        Args:
            obs: Observation dict (LeRobot format with ``task`` or ``prompt`` key).
            properties: Physical property estimates (from probe or ground truth).
            aggressiveness: Risk tolerance [0, 1].
            probe_mode: ``"probing"`` or ``"manipulation"``.
            training: If ``True``, apply dropout to conditioning tokens.

        Returns:
            Augmented observation dict (shallow copy).
        """
        obs = dict(obs)

        if not self.config.enabled:
            return obs

        props = properties or PropertyState.empty()
        agg = aggressiveness if aggressiveness is not None else self.config.default_aggressiveness
        mode = probe_mode or self.config.default_probe_mode

        task = obs.get("task", obs.get("prompt", "perform the task"))
        prompt = build_probenet_prompt(str(task), props, agg, mode)

        if training:
            prompt = drop_modalities(
                prompt,
                probe_mode_dropout=self.config.probe_mode_dropout,
                property_dropout=self.config.property_dropout,
                aggressiveness_dropout=self.config.aggressiveness_dropout,
                language_dropout=self.config.language_dropout,
            )

        obs["prompt"] = prompt
        return obs

    def embed_properties_in_frame(
        self,
        frame: dict,
        properties: PropertyState,
        aggressiveness: float | None = None,
        probe_mode: str = "manipulation",
    ) -> dict:
        """Embed property info into a LeRobot dataset frame.

        Called during data collection to annotate frames with ground truth
        physical property information in the ``task`` field.
        """
        frame = dict(frame)
        agg = aggressiveness if aggressiveness is not None else self.config.default_aggressiveness
        task = frame.get("task", "perform the task")
        prompt = build_probenet_prompt(str(task), properties, agg, probe_mode)
        frame["task"] = prompt
        return frame
