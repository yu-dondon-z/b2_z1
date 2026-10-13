"""VB10 actual contact/clearance telemetry, using the same loaded dynamics."""
import torch
from isaaclab.utils.math import quat_apply
from .b2z1_loaded_gun_env import B2Z1LoadedGunAMPEnv


class B2Z1GunClearanceAMPEnv(B2Z1LoadedGunAMPEnv):
    def step(self, action):
        result = super().step(action)
        robot = self.scene['robot']
        gun = self._gun_body_id
        sensor = self.scene['contact_forces']
        if not hasattr(self, '_vb10_gun_sensor_id'):
            self._vb10_gun_sensor_id = sensor.find_bodies('link_gun')[0][0]
        force = sensor.data.net_forces_w[:, self._vb10_gun_sensor_id]
        norm = torch.linalg.vector_norm(force,dim=-1)
        q = robot.data.body_link_quat_w[:,gun]
        point = torch.tensor([.003,-.032,-.05],device=self.device).expand(self.num_envs,3)
        axis = torch.tensor([0.,0.,1.],device=self.device).expand(self.num_envs,3)
        center = robot.data.body_link_pos_w[:,gun] + quat_apply(q,point)
        z_axis = quat_apply(q,axis)[:,2]
        # Conservative analytic lower bound for the flat convex mesh. Not the
        # old capsule: no hemisphere beyond the requested cylinder's length.
        bottom = center[:,2] - .55*z_axis.abs() - .028061*(1-z_axis.square()).clamp_min(0).sqrt()
        self.extras.setdefault('log',{}).update({
            'VB10/gun_contact_fraction': (norm>1.).float().mean(),
            'VB10/gun_contact_force_mean_n': norm.mean(),
            'VB10/gun_vertical_support_mean_n': force[:,2].clamp_min(0).mean(),
            'VB10/gun_bottom_clearance_mean_m': (bottom-self.scene.env_origins[:,2]).mean(),
        })
        # Readable to diagnostic scripts without patching the control loop.
        self.gun_contact_force_w = force
        self.gun_bottom_clearance_m = bottom-self.scene.env_origins[:,2]
        return result
