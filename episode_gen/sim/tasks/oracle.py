import numpy as np

_REST_POSE = np.array([0.0, -1.745, 1.571, 0.873, 0.0, 0.0], dtype=np.float32)
_GRIPPER_OPEN = 1.0
_GRIPPER_CLOSED = 0.0

PHASES = [
    ("approach", 200, np.array([0.0, -0.8, 1.0, 1.0, 1.58, _GRIPPER_OPEN])),
    ("grasp", 50, np.array([0.0, -0.8, 1.0, 1.0, 1.58, _GRIPPER_CLOSED])),
    ("lift", 100, np.array([0.0, -1.2, 1.0, 1.5, 1.58, _GRIPPER_CLOSED])),
    ("place", 200, np.array([0.2, -1.0, 1.2, 1.0, 1.58, _GRIPPER_OPEN])),
    ("release", 100, np.array([0.0, -1.0, 1.2, 1.0, 1.58, _GRIPPER_OPEN])),
]


class PickPlaceStateMachine:
    def __init__(self):
        self._step_count = 0
        self._episode_done = False
        self._phase_boundaries: list[tuple[int, int]] = []
        self._total_steps = sum(d for _, d, _ in PHASES)

    def setup(self, env) -> None:
        pass

    def check_success(self, env) -> bool:
        return True

    def get_action(self, env):
        cumulative = 0
        for _, duration, target in PHASES:
            if cumulative <= self._step_count < cumulative + duration:
                return target.copy()
            cumulative += duration
        return _REST_POSE.copy()

    def advance(self) -> None:
        self._step_count += 1
        if self._step_count >= self._total_steps:
            self._episode_done = True

    def reset(self) -> None:
        self._step_count = 0
        self._episode_done = False
        self._phase_boundaries = []

    @property
    def is_episode_done(self) -> bool:
        return self._episode_done
