"""Check actual nominal PhysX properties against the rebuilt mount75 inertia."""
import json
from .vd13_contract import ASSET


def audit_vd13_runtime_asset(robot):
    import torch
    manifest = json.loads(ASSET.with_suffix('.manifest.json').read_text())
    from .vd13_contract import TCP_POSITION
    from b2_z1.tasks.manager_based.b2_z1.mdp.tcp_frame import tcp_transform_for_asset
    names = list(robot.body_names)
    if len(names) != 24 or robot.num_joints != 18:
        raise RuntimeError('VD13 requires the preserved 24-body/18-joint articulation')
    mass, inertia = robot.data.default_mass.detach().cpu(), robot.data.default_inertia.detach().cpu()
    link, base = names.index('link06'), names.index('base_link')
    com = robot.data.body_com_pos_b.detach().cpu()
    expected_com = com.new_tensor(manifest['link06_combined_com_m'])
    if not torch.allclose(com[:, link], expected_com.expand_as(com[:, link]), atol=1e-5, rtol=0):
        raise RuntimeError('VD13 actual PhysX centre of mass differs from rebuilt flange+pipe')
    if tcp_transform_for_asset(robot) != (TCP_POSITION, (1., 0., 0., 0.)):
        raise RuntimeError('VD13 actual spawned asset and TCP selector disagree')
    expected = inertia.new_tensor(manifest['link06_combined_inertia_kg_m2']).reshape(9)
    if not torch.allclose(mass.sum(-1), mass.new_full((len(mass),), 81.9290202), atol=1e-3, rtol=0):
        raise RuntimeError('VD13 nominal total mass differs')
    if not torch.allclose(mass[:, link], mass.new_full((len(mass),), 3.2187581), atol=1e-4, rtol=0):
        raise RuntimeError('VD13 merged tube mass differs')
    if not torch.allclose(inertia[:, link], expected.expand_as(inertia[:, link]), atol=2e-4, rtol=0):
        raise RuntimeError('VD13 actual PhysX inertia differs from rebuilt tube+flange tensor')
    if not (torch.isfinite(inertia).all() and (torch.linalg.eigvalsh(inertia.reshape(-1, 3, 3)) > 0).all()):
        raise RuntimeError('VD13 invalid actual inertia')
    delta = robot.root_physx_view.get_masses().detach().cpu()-mass
    others = [i for i in range(len(names)) if i != base]
    if not torch.allclose(delta[:, others], torch.zeros_like(delta[:, others]), atol=1e-5, rtol=0):
        raise RuntimeError('VD13 non-base mass randomization is not allowed')
    if not ((delta[:, base] >= -1.001) & (delta[:, base] <= 3.001)).all():
        raise RuntimeError('VD13 base randomization outside inherited [-1,3]kg')
    return dict(version='VD13', status='passed', gun_mass_kg=2.93,
                nominal_total_mass_kg=float(mass[0].sum()), link06_inertia=inertia[0, link].tolist(),
                link06_com_m=com[0, link].tolist(), tcp_position_m=list(TCP_POSITION))
