"""Inspect imported VB10 geometry/mass and verify actual obstacle contact."""
import argparse
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args();launcher=AppLauncher(args)

import gymnasium as gym
import torch
from pxr import Usd,UsdGeom,UsdPhysics
import isaaclab.sim as sim_utils
from isaaclab.utils.math import quat_apply
from isaaclab_tasks.utils import parse_env_cfg
import b2_z1.tasks

def main():
    task='Unitree-b2z1-AMP-VB10-Clearance-Play'
    cfg=parse_env_cfg(task,device=args.device,num_envs=1)
    cfg.commands.ee_target.debug_vis=False
    cfg.commands.base_velocity.debug_vis=False
    env=gym.make(task,cfg=cfg).unwrapped
    try:
        robot=env.scene['robot'];gun=robot.find_bodies('link_gun')[0][0]
        mass=float(robot.root_physx_view.get_masses()[0,gun])
        assert abs(mass-3.25904698235386)<.01,('Gun mass changed',mass)
        stage=env.sim.stage
        collision=stage.GetPrimAtPath('/World/envs/env_0/Robot/link_gun/collisions')
        proto=collision.GetPrototype() if collision.IsInstance() else collision
        shapes=[p for p in Usd.PrimRange(proto) if p.HasAPI(UsdPhysics.CollisionAPI)]
        assert len(shapes)==1 and shapes[0].IsA(UsdGeom.Mesh),'Gun must import as one mesh, not capsule'
        assert UsdPhysics.CollisionAPI(shapes[0]).GetCollisionEnabledAttr().Get()
        points=torch.tensor([list(p) for p in UsdGeom.Mesh(shapes[0]).GetPointsAttr().Get()])
        extent=points.max(0).values-points.min(0).values
        assert abs(float(extent[2])-1.1)<1e-4,extent
        assert .056<=float(extent[0])<=.0562,extent
        center=robot.data.body_link_pos_w[:,gun]+quat_apply(
            robot.data.body_link_quat_w[:,gun],torch.tensor([[.003,-.032,-.05]],device=env.device))
        probe=sim_utils.SphereCfg(radius=.035,collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True))
        probe.func('/World/VB10CollisionProbe',probe,translation=center[0].cpu().tolist())
        peak=0.
        with torch.inference_mode():
            for _ in range(20):
                env.step(torch.zeros(env.action_space.shape,device=env.device))
                peak=max(peak,float(torch.linalg.vector_norm(env.gun_contact_force_w[0])))
        assert peak>1.,('No actual gun/probe contact',peak)
        print(f'VB10_COLLISION_PASS mass_kg={mass:.6f} extent_m={extent.tolist()} actual_probe_contact_n={peak:.3f}',flush=True)
    finally:env.close()

if __name__=='__main__':
    try:
        main()
    except Exception:
        import os,sys,traceback
        traceback.print_exc();sys.stdout.flush();sys.stderr.flush()
        # Kit shutdown may otherwise mask the assertion failure with exit 0.
        os._exit(1)
    else:
        launcher.app.close()
