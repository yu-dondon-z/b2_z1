"""Fail fast on actual PhysX properties of the separate 2.93 kg tool asset."""

from __future__ import annotations


def audit_vd10_runtime_asset(robot):
    import torch

    names = list(robot.body_names)
    if len(names) != 24 or robot.num_joints != 18 or "link06" not in names:
        raise RuntimeError(f"VD10 requires 24 bodies / 18 joints, got {len(names)} / {robot.num_joints}")
    if {"link_vd10_gun", "link_gun", "dummy_target"}.intersection(names):
        raise RuntimeError("VD10 must have one physical link06, with no separate zero-mass gun/TCP bodies")
    mass = robot.data.default_mass.detach().cpu()
    inertia = robot.data.default_inertia.detach().cpu()
    if not (torch.isfinite(mass).all() and torch.isfinite(inertia).all() and (mass > 0).all()):
        raise RuntimeError("VD10 nominal rigid-body mass/inertia is invalid")
    link_id, base_id = names.index("link06"), names.index("base_link")
    if not torch.allclose(mass.sum(-1), mass.new_full((mass.shape[0],), 81.9290202), atol=1e-3, rtol=0):
        raise RuntimeError("VD10 total mass differs from 78.9990202 + 2.93 kg")
    if not torch.allclose(mass[:, link_id], mass.new_full((mass.shape[0],), 3.2187581), atol=1e-4, rtol=0):
        raise RuntimeError("VD10 link06 must include the full 2.93 kg hollow tube")
    expected = inertia.new_tensor([
        .005026923864946012, -.00001708046959736788, -.010687389910048882,
        -.00001708046959736788, .34159733897570543, -.0000045261215408060674,
        -.010687389910048882, -.0000045261215408060674, .33889193093571174,
    ])
    if not torch.allclose(inertia[:, link_id], expected.expand_as(inertia[:, link_id]), atol=2e-4, rtol=0):
        raise RuntimeError("VD10 link06 inertia differs from the parallel-axis tube+flange calculation")
    if not (torch.linalg.eigvalsh(inertia.reshape(*inertia.shape[:-1], 3, 3)) > 0).all():
        raise RuntimeError("VD10 PhysX inertia must be positive definite")
    live = robot.root_physx_view.get_masses().detach().cpu()
    delta = live - mass
    others = [i for i in range(len(names)) if i != base_id]
    if not torch.allclose(delta[:, others], torch.zeros_like(delta[:, others]), atol=1e-5, rtol=0):
        raise RuntimeError("VD10 mass randomization changed a non-base link")
    base_delta = delta[:, base_id]
    if not ((base_delta >= -1.001) & (base_delta <= 3.001)).all():
        raise RuntimeError("VD10 base mass randomization exceeded [-1,3] kg")
    from b2_z1.tasks.manager_based.b2_z1.mdp.tcp_frame import tcp_pose_w, link_pose_w, quat_apply

    tcp_w, _ = tcp_pose_w(robot)
    link_w, link_quat_w = link_pose_w(robot, link_id)
    axis_w = quat_apply(link_quat_w, link_w.new_tensor((1.0, 0.0, 0.0)).expand_as(link_w))
    # Radius matters: a low TCP can collide even if its center is above ground.
    lower_end_w = tcp_w - 1.1 * axis_w
    lowest_tube_z = torch.minimum(tcp_w[:, 2], lower_end_w[:, 2]) - 0.028
    return {
        "version": "VD10", "status": "passed", "body_count": len(names), "joint_count": int(robot.num_joints),
        "gun_mass_kg": 2.93, "link06_mass_kg": float(mass[0, link_id]),
        "nominal_total_mass_kg": float(mass[0].sum()),
        "link06_inertia": inertia[0, link_id].tolist(),
        "base_mass_randomization_range_kg": [float(base_delta.min()), float(base_delta.max())],
        "initial_tcp_world_xyz_m": tcp_w[0].detach().cpu().tolist(),
        "initial_tube_min_surface_z_m": float(lowest_tube_z.min().item()),
        "initial_link06_world_xyz_m": link_w[0].detach().cpu().tolist(),
    }
