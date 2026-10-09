"""VB7: long load-robustness continuation; light payload is not gun removal."""
import torch
from isaaclab.utils import configclass
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from .b2_z1_vb4_env_cfg import B2Z1VB4PayoutAMPEnvCfg


def randomize_gun_mass_inertia(env, env_ids, asset_cfg, mass_range=(0.5, 4.2), light_fraction=0.25):
    """Startup domains: keep positive mass and proportionally scale inertia.

    Collision geometry and COM location remain unchanged. Light domains are
    a load approximation, not a claim to simulate a physically removed tool.
    """
    robot = env.scene[asset_cfg.name]
    ids = torch.arange(env.num_envs, device="cpu") if env_ids is None else env_ids.cpu()
    body = robot.find_bodies("link_gun")[0][0]
    mass = torch.empty(len(ids)).uniform_(*mass_range)
    light = torch.rand(len(ids)) < light_fraction
    mass[light] = 0.05
    masses = robot.root_physx_view.get_masses()
    inertias = robot.root_physx_view.get_inertias()
    ratio = mass / robot.data.default_mass[ids, body].cpu()
    masses[ids, body] = mass
    inertias[ids, body] = robot.data.default_inertia[ids, body].cpu() * ratio[:, None]
    robot.root_physx_view.set_masses(masses, ids)
    robot.root_physx_view.set_inertias(inertias, ids)
    env._vb7_light_payload_mask = light.to(env.device)
    env._vb7_gun_mass_kg = mass.to(env.device)
    print(f"[VB7 payload] gun mass {mass.min():.3f}–{mass.max():.3f} kg; light domains {light.sum()}/{len(ids)}; inertia scaled", flush=True)


@configclass
class B2Z1VB7LoadRobustAMPEnvCfg(B2Z1VB4PayoutAMPEnvCfg):
    hose_zero_load_probability: float = 0.25
    hose_drag_range_ns_per_m: tuple[float, float] = (0.0, 6.0)

    def __post_init__(self):
        super().__post_init__()
        self.hose_mass_range_kg = (0.0, 6.0)
        self.events.gun_payload = EventTermCfg(
            func=randomize_gun_mass_inertia, mode="startup",
            params={"asset_cfg": SceneEntityCfg("robot", body_names="link_gun"),
                    "mass_range": (0.5, 4.2), "light_fraction": 0.25},
        )
        self.commands.ee_target.tool_height_m = 0.01
        self.commands.ee_target.tool_height_range_m = (0.01, 0.02)
        self.commands.base_velocity.debug_vis = False
        self.commands.ee_target.debug_vis = False
