"""Train-only URDF posture reference. Never applied to simulation actuators/export.

The FK/Jacobian is evaluated in base_link coordinates and uses the physical
gun axis, not the differently oriented dummy_target frame. A masked auxiliary
loss guides the student; the student's 18 actions still drive every rollout.
"""
import math
import xml.etree.ElementTree as ET
import torch
from torch import nn


def rotation(axis, angle):
    axis = axis / torch.linalg.vector_norm(axis)
    x, y, z = axis.unbind()
    zero = x * 0
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero)).reshape(3, 3)
    eye = torch.eye(3, device=angle.device, dtype=angle.dtype)
    return eye + torch.sin(angle)[:, None, None] * skew + (1-torch.cos(angle))[:, None, None] * (skew @ skew)


class PoseReferenceTeacher(nn.Module):
    arm_names = [f"joint{i}" for i in range(1, 7)]

    def __init__(self, actor, urdf_path, arm_ids, default_arm, warmup_iterations=200,
                 margin_rad=.12, action_scale=.25, action_clip=5., ik_steps=6):
        super().__init__()
        self.actor = actor.eval().requires_grad_(False)
        self.arm_ids = list(arm_ids)
        if len(self.arm_ids) != 6 or len(set(self.arm_ids)) != 6:
            raise ValueError("Expected six unique articulation action rows")
        if warmup_iterations < 1 or ik_steps < 1 or margin_rad <= 0:
            raise ValueError("Invalid guide schedule/IK settings")
        if not all(math.isfinite(v) and v > 0 for v in (margin_rad, action_scale, action_clip)):
            raise ValueError('Guide margins/action bounds must be finite and positive')
        self.warmup_iterations, self.ik_steps = warmup_iterations, ik_steps
        self.iteration = 0
        self.action_scale = action_scale
        self.register_buffer('default_arm', torch.tensor(default_arm, dtype=torch.float32))
        root = ET.parse(urdf_path).getroot()
        bychild = {j.find('child').get('link'): j for j in root.findall('joint')}
        chain, child = [], 'dummy_target'
        while child != 'base_link':
            joint = bychild[child]; chain.append(joint); child = joint.find('parent').get('link')
        chain.reverse()
        self.chain = []
        lows, highs = {}, {}
        for i, joint in enumerate(chain):
            attrs = joint.find('origin').attrib if joint.find('origin') is not None else {}
            xyz = torch.tensor([float(x) for x in attrs.get('xyz', '0 0 0').split()])
            rpy = [float(x) for x in attrs.get('rpy', '0 0 0').split()]
            R = torch.eye(3)
            for axis, angle in zip(([0.,0.,1.], [0.,1.,0.], [1.,0.,0.]), rpy[::-1]):
                R = R @ rotation(torch.tensor(axis), torch.tensor([angle]))[0]
            self.register_buffer(f'origin_R_{i}', R)
            self.register_buffer(f'origin_p_{i}', xyz)
            name = joint.get('name')
            index = self.arm_names.index(name) if name in self.arm_names else None
            if index is not None:
                axis = torch.tensor([float(x) for x in joint.find('axis').get('xyz').split()])
                self.register_buffer(f'axis_{i}', axis)
                limits = joint.find('limit')
                lows[name], highs[name] = float(limits.get('lower'))+margin_rad, float(limits.get('upper'))-margin_rad
            elif joint.get('type') != 'fixed':
                raise ValueError(f'Unexpected moving joint in gun chain: {name}')
            self.chain.append((i, name, index))
        lower = torch.tensor([lows[n] for n in self.arm_names])
        upper = torch.tensor([highs[n] for n in self.arm_names])
        # Respect both physical margins and the existing bounded actor contract.
        lower = torch.maximum(lower, self.default_arm-action_scale*action_clip+.02)
        upper = torch.minimum(upper, self.default_arm+action_scale*action_clip-.02)
        if torch.any(lower >= upper):
            raise ValueError('No feasible guide range within physical/action limits')
        self.register_buffer('lower', lower)
        self.register_buffer('upper', upper)
        self.register_buffer('seed', torch.tensor([0., 2.7673, -2.4606, -.3067, 0., 0.]))
        self.last_mask = None
        self.last_null_projection = None

    def set_iteration(self, iteration):
        self.iteration = max(0, int(iteration))

    def fk(self, q):
        n = len(q)
        R = torch.eye(3, device=q.device, dtype=q.dtype).expand(n, 3, 3)
        p = torch.zeros(n, 3, device=q.device, dtype=q.dtype)
        axes, origins, gun_axis = [], [], None
        for i, name, index in self.chain:
            p = p + torch.einsum('nij,j->ni', R, getattr(self, f'origin_p_{i}'))
            R = R @ getattr(self, f'origin_R_{i}')
            if index is not None:
                axis = getattr(self, f'axis_{i}')
                axes.append(torch.einsum('nij,j->ni', R, axis)); origins.append(p)
                R = R @ rotation(axis, q[:, index])
            if name == 'joint_gun_mounter':
                gun_axis = R[:, :, 2]
        if gun_axis is None or len(axes) != 6:
            raise RuntimeError('Gun chain identity mismatch')
        axes, origins = torch.stack(axes, dim=2), torch.stack(origins, dim=2)
        Jp = torch.cross(axes, p.unsqueeze(-1)-origins, dim=1)
        Jz = torch.cross(axes, gun_axis.unsqueeze(-1).expand_as(axes), dim=1)
        return p, gun_axis, torch.cat((Jp, Jz), dim=1)

    @torch.no_grad()
    def solve(self, target, up):
        q = self.seed.expand(len(target), 6).clone().clamp(self.lower, self.upper)
        eye = torch.eye(6, device=q.device, dtype=q.dtype).unsqueeze(0)
        scale = torch.tensor([1.,1.,1.,.45,.45,.45], device=q.device)
        for _ in range(self.ik_steps):
            p, z, J = self.fk(q)
            residual = torch.cat((target-p, up-z), dim=1) * scale
            J = J * scale[None,:,None]
            delta = (J.transpose(1,2) @ torch.linalg.solve(
                J @ J.transpose(1,2) + .025**2 * eye, residual.unsqueeze(-1))).squeeze(-1)
            # A clipped unconstrained step can be worse than the seed near
            # the wrist/shoulder bounds. Keep only improving bounded steps.
            best_cost = residual.square().sum(dim=1)
            best_q = q
            for factor in (1., .10):
                trial = (q+factor*delta.clamp(-.15,.15)).clamp(self.lower,self.upper)
                tp, tz, _ = self.fk(trial)
                trial_cost = (torch.cat((target-tp,up-tz),dim=1)*scale).square().sum(dim=1)
                improved = trial_cost < best_cost
                best_q = torch.where(improved[:,None],trial,best_q)
                best_cost = torch.where(improved,trial_cost,best_cost)
            q = best_q
        p, z, _ = self.fk(q)
        error = torch.linalg.vector_norm(p-target, dim=1)
        axis_error = torch.linalg.vector_norm(z-up, dim=1)
        return q, error, axis_error

    @torch.no_grad()
    def forward(self, obs):
        if obs.ndim != 2 or obs.shape[1] != 70:
            raise ValueError('VB9 guide requires unnormalized 70-component observations')
        safe_obs = torch.nan_to_num(obs, nan=0., posinf=0., neginf=0.)
        prior = self.actor(safe_obs)
        if prior.shape[1] != 18:
            raise ValueError('VB9 guide requires an 18-action student')
        gravity = safe_obs[:,13:16]
        norm = torch.linalg.vector_norm(gravity, dim=1)
        up = -gravity / norm.clamp_min(1.e-6).unsqueeze(1)
        target = safe_obs[:,3:6]
        q, position_error, axis_error = self.solve(target, up)
        valid = torch.isfinite(obs).all(dim=1) & (norm > .8) & (norm < 1.2)
        valid &= (target[:,0] >= -.90) & (target[:,0] <= -.62)
        valid &= (position_error < .015) & (axis_error < .08)
        valid &= torch.isfinite(q).all(dim=1) & (up[:,2] > math.cos(.15))
        fraction = min(self.iteration/self.warmup_iterations, 1.)
        blend = .10 + .90*fraction*fraction*(3-2*fraction)
        self.last_mask = valid.float()*blend
        # Reference shaping must not oppose the student's task-space or
        # gravity compensation targets. Project primarily into the current
        # translational TCP nullspace; leave a weak 5% full-pose anchor.
        measured = safe_obs[:,16:34][:,self.arm_ids] + self.default_arm
        _, _, measured_J = self.fk(measured)
        Jp = measured_J[:,:3,:]
        eye3 = torch.eye(3,device=obs.device,dtype=obs.dtype).unsqueeze(0)
        eye6 = torch.eye(6,device=obs.device,dtype=obs.dtype).unsqueeze(0)
        self.last_null_projection = eye6 - Jp.transpose(1,2) @ torch.linalg.solve(
            Jp @ Jp.transpose(1,2)+.05**2*eye3,Jp)
        result = prior.clone()
        desired = (q-self.default_arm)/self.action_scale
        result[:,self.arm_ids] = torch.where(valid[:,None], desired, prior[:,self.arm_ids])
        return result

    def action_error_cost(self, delta):
        """Differentiable student loss; the cached geometry is frozen."""
        if self.last_null_projection is None or len(delta)!=len(self.last_null_projection):
            raise RuntimeError('Posture loss must use its matching teacher batch')
        projected = (self.last_null_projection @ delta.unsqueeze(-1)).squeeze(-1)
        return projected.square().mean(dim=1)+.05*delta.square().mean(dim=1)

    def guidance_mask(self, obs):
        if self.last_mask is None or len(self.last_mask) != len(obs):
            raise RuntimeError('Guide mask must match its preceding observation batch')
        return self.last_mask
