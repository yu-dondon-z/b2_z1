"""VB11: real-load standby, measured release and no intermittent ground support."""
import copy
import torch
from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm, TerminationTermCfg as DoneTerm, EventTermCfg, SceneEntityCfg
from .b2_z1_vb10_env_cfg import (B2Z1VB10ClearanceAMPEnvCfg, VB10ClearanceRewardsCfg,
                               B2Z1VB10ClearanceAMPEnvCfg_Play, B2Z1VB10ClearanceVisionAMPEnvCfg_Play)
from .mdp.commands.vb11_ready_command import VB11ReadyTrajectoryCommandCfg, VB11ReadyTrajectoryCommand
from .mdp import vb11_tool_control as tool


def assign_payload_domains(env, env_ids, asset_cfg, nominal_fraction=.50, light_fraction=.10):
    """Exact environment quotas at startup, positive masses and scaled inertia.

    Half the environments retain the URDF gun mass AND get 3kg hose on every
    reset. Light domains are still present, not silently removed from training.
    """
    if not 0 <= nominal_fraction <= 1 or not 0 <= light_fraction <= 1 - nominal_fraction:
        raise ValueError("Invalid VB11 payload quotas")
    robot = env.scene[asset_cfg.name]
    ids = torch.arange(env.num_envs) if env_ids is None else env_ids.cpu()
    body = robot.find_bodies("link_gun")[0][0]
    order = torch.randperm(len(ids))
    nominal = torch.zeros(len(ids), dtype=torch.bool)
    light = torch.zeros_like(nominal)
    n = round(len(ids) * nominal_fraction)
    nominal[order[:n]] = True
    light[order[n:n + round(len(ids) * light_fraction)]] = True
    mass = torch.empty(len(ids)).uniform_(.5, 4.2)
    default = robot.data.default_mass[ids, body].cpu()
    mass[nominal] = default[nominal]
    mass[light] = .05
    masses = robot.root_physx_view.get_masses()
    inertias = robot.root_physx_view.get_inertias()
    masses[ids, body] = mass
    inertias[ids, body] = robot.data.default_inertia[ids, body].cpu() * (mass / default)[:, None]
    robot.root_physx_view.set_masses(masses, ids)
    robot.root_physx_view.set_inertias(inertias, ids)
    env._vb11_nominal_load_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    env._vb11_nominal_load_mask[ids.to(env.device)] = nominal.to(env.device)
    # Retain inherited load telemetry and zero-hose handling for light domains.
    env._vb7_light_payload_mask = torch.zeros_like(env._vb11_nominal_load_mask)
    env._vb7_light_payload_mask[ids.to(env.device)] = light.to(env.device)
    env._vb7_gun_mass_kg = robot.root_physx_view.get_masses()[:, body].to(env.device)
    print(f"[VB11 payload] nominal gun + 3kg hose {n}/{len(ids)}; light {light.sum()}/{len(ids)}", flush=True)


@configclass
class VB11ReadyRewardsCfg(VB10ClearanceRewardsCfg):
    tcp_height_shortfall = RewTerm(func=tool.tcp_height_shortfall, weight=-2.)
    tcp_height_precision = RewTerm(func=tool.tcp_height_precision, weight=1.5)
    tcp_below_road = RewTerm(func=tool.tcp_below_road, weight=-4.)
    gun_bottom_margin = RewTerm(func=tool.gun_bottom_margin, weight=-1.,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="link_gun")})


def configure_ready(cfg):
    command = VB11ReadyTrajectoryCommandCfg()
    for key, value in vars(cfg.commands.ee_target).items():
        setattr(command, key, copy.deepcopy(value))
    command.class_type = VB11ReadyTrajectoryCommand
    command.file_standby_before_motion = True
    command.file_standby_height_tolerance_m = .01
    command.file_target_height_filter_in_world_frame = True
    command.use_capture_pose = True
    cfg.commands.ee_target = command
    # The command term supplies explicit standby domains and preparatory holds.
    cfg.commands.base_velocity.rel_standing_envs = 0.
    sensor = SceneEntityCfg("contact_forces", body_names="link_gun")
    cfg.terminations.gun_sustained_contact = DoneTerm(func=tool.continuous_gun_contact,
        params={"sensor_cfg": sensor, "threshold_n": 10., "duration_s": .8, "grace_s": 4.})
    cfg.terminations.gun_contact_budget = DoneTerm(func=tool.intermittent_gun_contact,
        params={"sensor_cfg": sensor, "threshold_n": 5., "budget_s": .4, "grace_s": 4., "recovery_rate": .25})
    cfg.terminations.startup_not_ready = DoneTerm(func=tool.startup_not_ready, params={"deadline_s": 12.})


@configclass
class B2Z1VB11ReadyAMPEnvCfg(B2Z1VB10ClearanceAMPEnvCfg):
    rewards: VB11ReadyRewardsCfg = VB11ReadyRewardsCfg()
    vb9_arm_guide_weight: float = .08
    vb11_nominal_hose_mass_kg: float = 3.

    def __post_init__(self):
        super().__post_init__()
        configure_ready(self)
        self.hose_zero_load_probability = .15
        self.events.gun_payload = EventTermCfg(func=assign_payload_domains, mode="startup",
            params={"asset_cfg": SceneEntityCfg("robot", body_names="link_gun"),
                    "nominal_fraction": .50, "light_fraction": .10})
        self.events.vb9_pose_reset.params["fraction"] = .75
        self.rewards.gun_support_force.weight = -1.5
        self.rewards.undesired_contacts.weight = -6.
        # Preserve AMP, gait/height/heading rewards and existing curve gates.


@configclass
class B2Z1VB11ReadyAMPEnvCfg_Play(B2Z1VB10ClearanceAMPEnvCfg_Play):
    def __post_init__(self):
        super().__post_init__()
        configure_ready(self)
        self.commands.ee_target.vb11_standby_fraction = 0.
        self.commands.ee_target.vb11_hold_range_s = (4., 4.)


@configclass
class B2Z1VB11ReadyVisionAMPEnvCfg_Play(B2Z1VB10ClearanceVisionAMPEnvCfg_Play):
    def __post_init__(self):
        super().__post_init__()
        configure_ready(self)
