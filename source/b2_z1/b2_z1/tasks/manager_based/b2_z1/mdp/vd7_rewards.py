"""VD7 physical TCP continuity and worst-foot impact; reward truth only."""
import torch
from isaaclab.managers import ManagerTermBase
from .tcp_frame import tcp_pose_b
from .vd7_filter import filter_error


class TCPErrorHighpass(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.lowpass = torch.zeros(env.num_envs,3,device=env.device)
        self.valid = torch.zeros(env.num_envs,dtype=torch.bool,device=env.device)

    def reset(self, env_ids=None):
        self.valid[slice(None) if env_ids is None else env_ids] = False

    def __call__(self, env, command_name='ee_target', cutoff=3., deadzone=.002, scale=.01):
        current,_ = tcp_pose_b(env.scene['robot'])
        error = current-env.command_manager.get_command(command_name)[:,:3]
        active = env.episode_length_buf*env.step_dt > 2.
        penalty,self.lowpass,self.valid = filter_error(
            error,self.lowpass,self.valid,active,env.step_dt,cutoff,deadzone,scale)
        return penalty


def worst_foot_impact(env, sensor_cfg, threshold=350., scale=200.):
    count = max(1,round(env.step_dt/env.physics_dt))
    force = env.scene[sensor_cfg.name].data.net_forces_w_history[:, :count, sensor_cfg.body_ids, 2]
    per_foot = ((force.amax(dim=1)-threshold).clamp_min(0)/scale).square()
    return per_foot.amax(dim=-1)
