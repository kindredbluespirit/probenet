"""Smoke tests for the scripted probe runner."""


from probenet.env import SO101Env
from probenet.probe import ProbeRunner, default_probe_config, extract_probe_features


def test_probe_runner():
    """The probe should run through all phases and log a signal."""
    env = SO101Env(object_type="shell_a")
    env.reset(seed=0)
    runner = ProbeRunner(env)
    signal = runner.run()

    assert "actuator_force" in signal
    assert "qfrc_actuator" in signal
    assert "timestamps" in signal
    assert "phase_boundaries" in signal

    total_steps = sum(phase.duration_steps for phase in runner.phases)
    assert signal["actuator_force"].shape[0] == total_steps
    assert signal["qfrc_actuator"].shape[0] == total_steps
    assert len(signal["phase_names"]) == total_steps

    env.close()


def test_probe_features():
    """Feature extraction should return scalar statistics."""
    env = SO101Env(object_type="shell_a")
    env.reset(seed=0)
    runner = ProbeRunner(env)
    signal = runner.run()
    features = extract_probe_features(signal)
    for _key, value in features.items():
        assert isinstance(value, float)
    env.close()


def test_probe_default_config():
    """The default config should build the expected phases."""
    config = default_probe_config()
    phases = config.build_phases()
    phase_names = [phase.name for phase in phases]
    assert "approach" in phase_names
    assert "squeeze" in phase_names
    assert "lift" in phase_names
    assert "hold" in phase_names
    assert "release" in phase_names
