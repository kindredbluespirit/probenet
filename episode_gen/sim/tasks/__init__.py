import gymnasium as gym

gym.register(
    id="ProbeNet-SO101-PickPlace-Prim-v0",
    entry_point=f"{__name__}.pick_place_prim:PickPlacePrimEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pick_place_prim_cfg:PickPlacePrimEnvCfg",
    },
)
