"""VB8 costs measured at physical links, not post-policy motion filters."""
import math
import torch


def _huber_cost(value, scale):
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("stability scale must be finite and positive")
    return torch.sqrt(1.0 + (value / scale).square()) - 1.0


def base_world_heave_cost(env, asset_cfg, speed_scale=0.10):
    """Penalize actual world-z velocity, including pitch-induced link motion.

    Forward/lateral progress and commanded yaw are not penalized. On the
    flat-road VB task world-z is the road normal; not a rough-terrain cost.
    """
    asset = env.scene[asset_cfg.name]
    return _huber_cost(asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids[0], 2], speed_scale)


def gun_world_tilt_rate_cost(env, asset_cfg, speed_scale=0.50):
    """Discourage gun roll/pitch oscillation; leave world-yaw motion free."""
    asset = env.scene[asset_cfg.name]
    angular = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids[0], :2]
    return _huber_cost(angular, speed_scale).mean(dim=1)


def gun_vertical_axis_cost(env, asset_cfg):
    """Keep the cylinder's local z-axis vertical; free yaw and axis sign.

    The gun URDF cylinder is along local z. Its TCP position and original
    positive tracking rewards still matter; this bounded cost cannot replace
    them. This assumes a flat road and gravity along world -z.
    """
    asset = env.scene[asset_cfg.name]
    q = asset.data.body_link_quat_w[:, asset_cfg.body_ids[0]]
    axis_z_w = 1.0 - 2.0 * (q[:, 1].square() + q[:, 2].square())
    return (1.0 - axis_z_w.square()).clamp(0.0, 1.0)


def arm_limit_buffer_cost(env, asset_cfg, margin_rad=0.10,
                          action_name="JointPositionAction"):
    """Leave arm compensation room, penalizing measured and requested poses.

    Using requested targets too prevents the actor hiding a large command
    beyond a physical stop. This is a reward, not a hard target clamp or a
    changed joint limit. JointPositionAction resolves all joints in asset
    order; assert that mapping before indexing targets.
    """
    if not math.isfinite(margin_rad) or margin_rad <= 0:
        raise ValueError("arm limit margin must be finite and positive")
    asset = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids
    limits = asset.data.joint_pos_limits[:, ids]
    widths = limits[..., 1] - limits[..., 0]
    if torch.any(widths <= 2 * margin_rad):
        raise ValueError("arm limit buffer leaves no feasible joint range")
    lower, upper = limits[..., 0] + margin_rad, limits[..., 1] - margin_rad
    term = env.action_manager.get_term(action_name)
    # Isaac Lab 2.2 exposes the resolved actuator mapping as _joint_ids.
    term_ids = term._joint_ids
    if not (isinstance(term_ids, slice) and term_ids == slice(None)):
        expected = list(range(asset.data.joint_pos.shape[1]))
        if list(term_ids) != expected:
            raise ValueError("VB8 requires all-joint action in articulation order")
    actual = asset.data.joint_pos[:, ids]
    desired = term.processed_actions[:, ids]
    def cost(q):
        violation = (lower - q).clamp_min(0) + (q - upper).clamp_min(0)
        return _huber_cost(violation, margin_rad).mean(dim=1)
    return 0.5 * (cost(actual) + cost(desired))
