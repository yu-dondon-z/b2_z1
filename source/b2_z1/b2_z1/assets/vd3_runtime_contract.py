"""Fail-fast checks for the VD3 zero-gun-load PhysX articulation."""

from __future__ import annotations


def audit_vd3_runtime_asset(robot):
    """Validate nominal PhysX properties and startup mass randomization.

    IsaacLab records default_mass/default_inertia before startup events. Live
    base mass may differ by [-1, 3] kg, while all other masses remain unchanged.
    Returns JSON-serializable evidence for the training log.
    """
    import torch

    names = list(robot.body_names)
    if len(names) != 24 or robot.num_joints != 18:
        raise RuntimeError(f"VD3 requires 24 physical bodies / 18 joints, got {len(names)} / {robot.num_joints}.")
    forbidden = {"link_mounter", "link_gun", "dummy_target"}.intersection(names)
    if forbidden:
        raise RuntimeError(f"VD3 massless frames became physical rigid bodies: {sorted(forbidden)}.")
    mass = robot.data.default_mass.detach().cpu()
    inertia = robot.data.default_inertia.detach().cpu()
    if not torch.isfinite(mass).all() or not (mass > 0).all() or not torch.isfinite(inertia).all():
        raise RuntimeError("VD3 contains non-finite or non-positive nominal rigid-body mass/inertia.")
    total = mass.sum(-1)
    if not torch.allclose(total, total.new_full(total.shape, 78.9990202), atol=1.e-3, rtol=0.):
        raise RuntimeError(f"VD3 nominal total mass must be 78.9990202 kg; observed {total.min()}..{total.max()}.")
    end_id, base_id = names.index("link06"), names.index("base_link")
    if not torch.allclose(mass[:, end_id], mass.new_full(mass[:, end_id].shape, .2887581), atol=1.e-6, rtol=0.):
        raise RuntimeError("VD3 link06 mass includes unintended gun or dummy payload.")
    expected = inertia.new_tensor([
        .0001833, .0000012, .0000005,
        .0000012, .0001475, .0000001,
        .0000005, .0000001, .0001468,
    ])
    if not torch.allclose(inertia[:, end_id], expected.expand_as(inertia[:, end_id]), atol=1.e-7, rtol=0.):
        raise RuntimeError("VD3 link06 inertia differs from its original inertial tensor.")
    eigenvalues = torch.linalg.eigvalsh(inertia.reshape(*inertia.shape[:-1], 3, 3))
    if not (eigenvalues > 0).all():
        raise RuntimeError("VD3 contains a non-positive-definite nominal rigid-body inertia tensor.")
    live = robot.root_physx_view.get_masses().detach().cpu()
    delta = live - mass
    nonbase = [index for index in range(len(names)) if index != base_id]
    if not torch.allclose(delta[:, nonbase], torch.zeros_like(delta[:, nonbase]), atol=1.e-5, rtol=0.):
        raise RuntimeError("VD3 startup mass randomization modified bodies other than base_link.")
    base_delta = delta[:, base_id]
    if not ((base_delta >= -1.001) & (base_delta <= 3.001)).all():
        raise RuntimeError("VD3 base startup mass randomization exceeded [-1, 3] kg.")
    return {
        "version": "VD3", "status": "passed", "body_count": len(names), "joint_count": int(robot.num_joints),
        "gun_added_mass_kg": 0., "gun_added_inertia": [0.]*9, "tcp_parent_body": "link06",
        "nominal_total_mass_kg": float(total[0]), "link06_mass_kg": float(mass[0, end_id]),
        "link06_inertia": inertia[0, end_id].tolist(),
        "live_total_mass_range_kg": [float(live.sum(-1).min()), float(live.sum(-1).max())],
        "base_mass_randomization_range_kg": [float(base_delta.min()), float(base_delta.max())],
        "body_names": names,
    }
