"""VB13 train-only static-load reference; NEVER writes actuator commands.

At the IK solution q*, PD needs q_des = q* + tau_gravity / kp to hold it.
Use the URDF arm/gun COMs and effective hanging hose at its connector.
The full student still produces every rollout action and exported action.
"""
import math
import xml.etree.ElementTree as ET
import torch
from .pose_reference_teacher import PoseReferenceTeacher, rotation


class GravityPoseReferenceTeacher(PoseReferenceTeacher):
    def __init__(self, *args, arm_stiffness, hose_mass_kg=0., max_bias_rad=.45, **kwargs):
        super().__init__(*args, **kwargs)
        kp = torch.as_tensor(arm_stiffness, dtype=torch.float32)
        if kp.shape != (6,) or not torch.isfinite(kp).all() or (kp <= 0).any():
            raise ValueError("VB13 needs the six actual positive arm PD stiffnesses")
        if not math.isfinite(hose_mass_kg) or hose_mass_kg < 0 or not 0 < max_bias_rad <= .5:
            raise ValueError("Invalid static-load reference bounds")
        self.register_buffer('kp', kp)
        self.hose_mass_kg, self.max_bias_rad = hose_mass_kg, max_bias_rad
        # args[1] is the same URDF used by parent FK, including flat collider.
        root = ET.parse(args[1]).getroot()
        joints = {j.get('name'): j for j in root.findall('joint')}
        links = {l.get('name'): l for l in root.findall('link')}
        self.com_entries = []
        for i, name, index in self.chain:
            child = joints[name].find('child').get('link')
            inertial = links[child].find('inertial')
            mass, com = 0., [0., 0., 0.]
            if inertial is not None:
                mass_node = inertial.find('mass')
                mass = max(0., float(mass_node.get('value'))) if mass_node is not None else 0.
                origin = inertial.find('origin')
                if origin is not None:
                    com = [float(v) for v in origin.get('xyz', '0 0 0').split()]
            self.register_buffer(f'com_{i}', torch.tensor(com))
            self.com_entries.append((i, child, mass))
        self.last_bias = None

    def gravity_torque(self, q, up):
        """Positive support torque dU/dq, also testable by finite differences."""
        n = len(q)
        R = torch.eye(3, dtype=q.dtype, device=q.device).expand(n, 3, 3)
        p = torch.zeros(n, 3, dtype=q.dtype, device=q.device)
        axes = torch.zeros(n, 3, 6, dtype=q.dtype, device=q.device)
        origins = torch.zeros_like(axes)
        active = torch.zeros(6, dtype=q.dtype, device=q.device)
        tau = torch.zeros_like(q)
        potential = torch.zeros(n, dtype=q.dtype, device=q.device)
        for (i, name, index), (_, child, mass) in zip(self.chain, self.com_entries):
            p = p + torch.einsum('nij,j->ni', R, getattr(self, f'origin_p_{i}'))
            R = R @ getattr(self, f'origin_R_{i}')
            if index is not None:
                axis = getattr(self, f'axis_{i}')
                axes[:, :, index] = torch.einsum('nij,j->ni', R, axis)
                origins[:, :, index] = p
                active[index] = 1.
                R = R @ rotation(axis, q[:, index])
            com = p + torch.einsum('nij,j->ni', R, getattr(self, f'com_{i}'))
            J = torch.cross(axes, com.unsqueeze(-1)-origins, dim=1) * active
            tau += mass * 9.8 * torch.einsum('nki,nk->ni', J, up)
            potential += mass * 9.8 * (com * up).sum(-1)
            if child == 'link_gun':
                point = p + torch.einsum('nij,j->ni', R, q.new_tensor([0., -.03, .48]))
                J = torch.cross(axes, point.unsqueeze(-1)-origins, dim=1) * active
                tau += self.hose_mass_kg * 9.8 * torch.einsum('nki,nk->ni', J, up)
                potential += self.hose_mass_kg * 9.8 * (point * up).sum(-1)
        return tau, potential

    @torch.no_grad()
    def forward(self, obs):
        result = super().forward(obs)
        up = -torch.nan_to_num(obs[:, 13:16])
        up = up / up.norm(dim=-1, keepdim=True).clamp_min(1.e-6)
        q_ik = result[:, self.arm_ids] * self.action_scale + self.default_arm
        tau, _ = self.gravity_torque(q_ik, up)
        bias = (tau / self.kp).clamp(-self.max_bias_rad, self.max_bias_rad)
        compensated = (q_ik + bias).clamp(self.lower, self.upper)
        valid = self.last_mask > 0
        result[:, self.arm_ids] = torch.where(valid[:, None],
            (compensated-self.default_arm)/self.action_scale, result[:, self.arm_ids])
        self.last_bias = (compensated-q_ik).detach()
        return result

    def action_error_cost(self, delta):
        # VB9 used mainly TCP-nullspace shaping. Acquisition needs a real
        # load-bearing pose reference, not only nullspace regularization.
        return delta.square().mean(dim=-1)
