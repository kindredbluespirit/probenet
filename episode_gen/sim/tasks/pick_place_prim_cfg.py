import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.shapes.shapes_cfg import CuboidCfg
from isaaclab.utils import configclass

from .robot_cfg import SO101_FOLLOWER_CFG


@configclass
class PickPlacePrimEnvCfg(DirectRLEnvCfg):
    decimation = 1
    episode_length_s = 15.0
    action_scale = 1.0
    action_space = 6
    observation_space = 6
    state_space = 0

    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 90.0,
        render_interval=1,
        disable_contact_processing=True,
        use_fabric=False,
    )

    robot: ArticulationCfg = SO101_FOLLOWER_CFG.replace(
        prim_path="/World/Robot"
    )

    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/Object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=sim_utils.CuboidCfg(size=(0.04, 0.04, 0.04)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.6, 0.35, 0.25),
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(2.2, -0.61, 0.4),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
    )

    front_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/Robot/base/front_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, -0.5, 0.6),
            rot=(0.1650476, -0.9862856, 0.0, 0.0),
            convention="ros",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=28.7,
            focus_distance=400.0,
            horizontal_aperture=38.11,
            clipping_range=(0.01, 50.0),
            lock_camera=True,
        ),
        width=640,
        height=480,
        update_period=1.0 / 30.0,
    )
