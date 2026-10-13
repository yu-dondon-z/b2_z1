"""VB10 flat-road contact constraints; no force or action modification."""
import math
import torch


def gun_support_force_cost(env, sensor_cfg, force_scale_n=20.):
    if not math.isfinite(force_scale_n) or force_scale_n <= 0:
        raise ValueError('Gun force scale must be positive and finite')
    force = env.scene[sensor_cfg.name].data.net_forces_w[:, sensor_cfg.body_ids[0]]
    # Magnitude also penalizes lateral scraping, not just vertical support.
    ratio = torch.linalg.vector_norm(force, dim=-1) / force_scale_n
    return (torch.sqrt(1+ratio.square())-1).clamp(max=10.)


def sustained_gun_contact(env, sensor_cfg, threshold_n=10., duration_s=.8, grace_s=4.):
    """Terminate continuous loaded gun contact after startup grace.

    Timer is per environment and cleared at reset (episode age <= grace), not
    shared by parallel robots. A single contact spike does not end an episode.
    """
    if not all(math.isfinite(v) and v > 0 for v in (threshold_n, duration_s)) or not math.isfinite(grace_s) or grace_s < 0:
        raise ValueError('Invalid gun contact thresholds')
    force = env.scene[sensor_cfg.name].data.net_forces_w[:, sensor_cfg.body_ids[0]]
    if not hasattr(env, '_vb10_gun_contact_elapsed'):
        env._vb10_gun_contact_elapsed = torch.zeros_like(env.episode_length_buf, dtype=torch.float32)
    age = env.episode_length_buf * env.step_dt
    contact = (torch.linalg.vector_norm(force, dim=-1) > threshold_n) & (age > grace_s)
    env._vb10_gun_contact_elapsed[:] = torch.where(
        contact, env._vb10_gun_contact_elapsed + env.step_dt, 0.)
    return env._vb10_gun_contact_elapsed >= duration_s
