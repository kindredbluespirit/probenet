"""Smoke tests for the SO-101 MuJoCo environment."""

import numpy as np

from probenet.env import SO101Env


def test_env_loads():
    """The environment should load and reset without error."""
    env = SO101Env(object_type="shell_a")
    obs, info = env.reset(seed=0)
    assert "image" in obs
    assert "state" in obs
    assert obs["image"].shape == (224, 224, 3)
    assert info["object_type"] == "shell_a"
    env.close()


def test_env_step():
    """The environment should accept a normalized action and step."""
    env = SO101Env(object_type="shell_a")
    env.reset(seed=0)
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs["image"].shape == (224, 224, 3)
    assert not terminated
    assert not truncated
    assert info["object_type"] == "shell_a"
    env.close()


def test_env_object_types():
    """Both shell types should be loadable."""
    for object_type in ("shell_a", "shell_b"):
        env = SO101Env(object_type=object_type)
        obs, info = env.reset(seed=0)
        assert info["object_type"] == object_type
        env.close()


def test_env_probe_signal():
    """The probe signal should have the expected actuator keys."""
    env = SO101Env(object_type="shell_a")
    env.reset(seed=0)
    signal = env.get_probe_signal()
    assert "actuator_force" in signal
    assert "qfrc_actuator" in signal
    assert signal["actuator_force"].shape == (env.model.nu,)
    assert signal["qfrc_actuator"].shape == (env.model.nv,)
    env.close()


def test_env_lerobot_obs():
    """LeRobot-compatible observation should have expected keys and types."""
    env = SO101Env(object_type="shell_a")
    env.reset(seed=0)

    # Step once so joints settle.
    env.step(np.zeros(6, dtype=np.float32))
    obs = env.get_lerobot_obs()

    assert "cam_primary" in obs
    assert obs["cam_primary"].shape == (224, 224, 3)

    for key in ("shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
                "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"):
        assert key in obs, f"missing key {key}"
        assert isinstance(obs[key], float), f"{key} is {type(obs[key])}"

    assert 0.0 <= obs["gripper.pos"] <= 1.0, f"gripper out of range: {obs['gripper.pos']}"
    env.close()


def test_env_pick_place_trajectory():
    """The pick-and-place trajectory should have the right shape."""
    env = SO101Env(object_type="shell_a")
    traj = env.pick_place_trajectory
    assert traj.ndim == 2
    assert traj.shape[1] == 6
    assert traj.shape[0] > 200
    env.close()
