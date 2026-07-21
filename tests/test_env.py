"""Tests for the IsaacSO101Env and LeRobot adapter."""

import sys

import numpy as np
import pytest

try:
    import isaacsim  # noqa: F401

    HAS_ISAAC_SIM = True
except ImportError:
    HAS_ISAAC_SIM = False

from probenet.env import JOINT_NAMES, IsaacSO101Env, LerobotAdapter, ObjectSpec
from probenet.env.isaac_env import _pack_joint_dict, _unpack_joint_dict


class DummyLeIsaacEnv:
    """Minimal stub to test IsaacSO101Env without Isaac Sim."""

    def __init__(self):
        self.device = "cpu"
        self.num_envs = 1
        self._step_count = 0
        self.scene = {}

    def reset(self):
        self._step_count = 0
        obs = self._make_obs()
        return obs, {}

    def step(self, action):
        self._step_count += 1
        obs = self._make_obs()
        done = self._step_count >= 500
        return obs, 0.0, done, False, {}

    def _get_observations(self):
        return self._make_obs()

    def _make_obs(self):
        return {
            "policy": {
                "joint_pos": np.zeros((1, 6), dtype=np.float32),
                "joint_vel": np.zeros((1, 6), dtype=np.float32),
                "front": np.zeros((1, 480, 640, 3), dtype=np.uint8),
                "wrist": np.zeros((1, 480, 640, 3), dtype=np.uint8),
            }
        }

    def close(self):
        pass

    def __getattr__(self, name):
        def _noop(*args, **kwargs):
            return None

        return _noop


@pytest.fixture
def dummy_env():
    return IsaacSO101Env(DummyLeIsaacEnv())


class TestJointPacking:
    def test_pack_joint_dict(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
        result = _pack_joint_dict("pos", values)
        assert result["shoulder_pan.pos"] == 1.0
        assert result["shoulder_lift.pos"] == 2.0
        assert result["gripper.pos"] == 6.0
        assert len(result) == 6

    def test_unpack_joint_dict(self):
        action = {
            "shoulder_pan.pos": 1.0,
            "shoulder_lift.pos": 2.0,
            "elbow_flex.pos": 3.0,
            "wrist_flex.pos": 4.0,
            "wrist_roll.pos": 5.0,
            "gripper.pos": 6.0,
        }
        result = _unpack_joint_dict(action)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])


class TestIsaacSO101Env:
    def test_reset(self, dummy_env):
        obs = dummy_env.reset()
        assert "shoulder_pan.pos" in obs or isinstance(obs, np.ndarray)

    def test_step(self, dummy_env):
        dummy_env.reset()
        obs, reward, done, info = dummy_env.step(np.zeros(6, dtype=np.float64))
        assert isinstance(obs, dict) or isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(done, bool)

    def test_lerobot_adapter(self, dummy_env):
        dummy_env.reset()
        adapter = LerobotAdapter(dummy_env)
        obs = adapter.get_observation()
        assert isinstance(obs, dict)

        adapter.send_action({
            "shoulder_pan.pos": 0.0,
            "shoulder_lift.pos": 0.0,
            "elbow_flex.pos": 0.0,
            "wrist_flex.pos": 0.0,
            "wrist_roll.pos": 0.0,
            "gripper.pos": 0.0,
        })

    def test_joint_names(self):
        assert len(JOINT_NAMES) == 6
        assert JOINT_NAMES[0] == "shoulder_pan"
        assert JOINT_NAMES[-1] == "gripper"

    def test_object_spec(self):
        spec = ObjectSpec(name="test", usd_path="test.usd", mass=0.5)
        assert spec.name == "test"
        assert spec.mass == 0.5


@pytest.mark.skipif(not HAS_ISAAC_SIM, reason="Isaac Sim not installed")
class TestIsaacSO101EnvLive:
    def test_live_env_requires_isaac_sim(self):
        pytest.skip("Live Isaac Sim tests not yet implemented (Phase 2)")
