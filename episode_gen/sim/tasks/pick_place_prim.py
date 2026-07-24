import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import TiledCamera

from .pick_place_prim_cfg import PickPlacePrimEnvCfg


def _joint_pos(env: DirectRLEnv) -> torch.Tensor:
    robot = env.scene["robot"]
    return robot.data.joint_pos[:, :6]


class PickPlacePrimEnv(DirectRLEnv):
    cfg: PickPlacePrimEnvCfg

    def __init__(self, cfg: PickPlacePrimEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self._object = RigidObject(self.cfg.object)
        self._front_camera = TiledCamera(self.cfg.front_camera)

        self.scene.articulations["robot"] = self._robot
        self.scene.rigid_objects["object"] = self._object
        self.scene.sensors["front"] = self._front_camera

        light_cfg = sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0)
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone() * self.cfg.action_scale

    def _apply_action(self) -> None:
        self.scene["robot"].set_joint_position_target(self.actions)

    def _get_observations(self) -> dict:
        obs = {
            "policy": {
                "joint_pos": _joint_pos(self),
                "actions": self.actions,
            }
        }
        return obs

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return time_out, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        super()._reset_idx(env_ids)

        joint_pos = self._robot.data.default_joint_pos[env_ids]
        self._robot.write_joint_position_to_sim(
            joint_pos, joint_ids=None, env_ids=env_ids
        )
        self._robot.write_joint_velocity_to_sim(
            torch.zeros_like(joint_pos), joint_ids=None, env_ids=env_ids
        )
