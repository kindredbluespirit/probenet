"""Tests for the ProbeRunner and signal extraction."""

import numpy as np

from probenet.probe.probe_runner import (
    ProbeConfig,
    ProbePhase,
    ProbeRunner,
    default_probe_config,
    extract_probe_features,
)


class TestProbeConfig:
    def test_default_config(self):
        config = default_probe_config()
        assert len(config.build_phases()) == 5
        assert config.approach_joints.shape == (6,)
        assert config.approach_joints[-1] == 0.0  # gripper open

    def test_build_phases(self):
        config = default_probe_config()
        phases = config.build_phases()
        assert phases[0].name == "approach"
        assert phases[1].name == "squeeze"
        assert phases[2].name == "lift"
        assert phases[3].name == "hold"
        assert phases[4].name == "release"
        assert phases[0].gripper == 0.0
        assert phases[1].gripper == 0.8

    def test_custom_config(self):
        ctrl = np.ones(6, dtype=np.float32)
        config = ProbeConfig(
            approach_joints=ctrl,
            squeeze_joints=ctrl,
            lift_joints=ctrl,
            approach_steps=10,
            squeeze_steps=20,
            lift_steps=30,
            hold_steps=40,
        )
        phases = config.build_phases()
        assert phases[0].duration_steps == 10
        assert phases[1].duration_steps == 20
        assert phases[2].duration_steps == 30
        assert phases[3].duration_steps == 40
        assert phases[4].duration_steps == 20  # release uses squeeze_steps


class TestProbeRunner:
    def test_run_with_dummy_env(self):
        class DummyEnv:
            def __init__(self):
                self.step_count = 0

            def step(self, action):
                self.step_count += 1
                return np.zeros(6), 0.0, False, {}

            def get_probe_signal(self):
                return {"actuator_force": np.random.randn(6).astype(np.float32)}

            def get_observation(self):
                return {f"{j}.pos": 0.0 for j in [
                    "shoulder_pan", "shoulder_lift", "elbow_flex",
                    "wrist_flex", "wrist_roll", "gripper",
                ]}

        env = DummyEnv()
        config = ProbeConfig(
            approach_joints=np.zeros(6, dtype=np.float32),
            squeeze_joints=np.ones(6, dtype=np.float32),
            lift_joints=np.ones(6, dtype=np.float32) * 2,
            approach_steps=10,
            squeeze_steps=10,
            lift_steps=5,
            hold_steps=5,
        )
        runner = ProbeRunner(env, config)
        signal = runner.run()

        total_steps = 10 + 10 + 5 + 5 + 10  # approach + squeeze + lift + hold + release
        assert signal["actuator_force"].shape == (total_steps, 6)
        assert signal["joint_pos"].shape == (total_steps, 6)
        assert signal["ctrl"].shape == (total_steps, 6)
        assert len(signal["phase_names"]) == total_steps
        assert len(signal["phase_boundaries"]) == 5

        # Verify phase names
        assert signal["phase_names"][0] == "approach"
        assert signal["phase_names"][-1] == "release"


class TestSignalFeatures:
    def test_extract_features(self):
        signal = {
            "actuator_force": np.random.randn(100, 6).astype(np.float32),
        }
        features = extract_probe_features(signal)
        assert "af_mean" in features
        assert "af_max" in features
        assert "af_std" in features
        assert isinstance(features["af_mean"], float)
