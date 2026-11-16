#!/usr/bin/env python3
"""Offline posture/motion review of existing first-episode VD13/VD14 logs.

No policy, physics, robot IO or writes to the source logs. Compare policies
within one simulator/reset/path, not Isaac's randomized world-Y with a straight
MuJoCo run. Contact force phases differ and are explicitly aligned below.
"""
import argparse
import gzip
import importlib.util
import json
from pathlib import Path
import tempfile

import numpy as np

from analyze_vd12_play import Settings, filtered_metrics, legacy_metrics, sha256, touchdowns, vector_rms
from vd14_play_contract import URDF, ASSETS


def distribution(a):
    a = np.asarray(a)
    return None if not a.size else dict(zip(
        ('min', 'p05', 'median', 'p95', 'max'), map(float, np.percentile(a, [0, 5, 50, 95, 100]))))


def rotation_wxyz(quaternion):
    q = np.asarray(quaternion, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1) > .001:
        raise ValueError('Expected finite unit wxyz quaternion')
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)),
                     (2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)),
                     (2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y))))


def posture(foot_b, hip_b, rotations):
    foot_b, hip_b, rotations = (np.asarray(a, dtype=float) for a in (foot_b, hip_b, rotations))
    count = len(foot_b)
    if (count == 0 or foot_b.shape != (count, 4, 3) or hip_b.shape != (count, 4, 3)
            or rotations.shape != (count, 3, 3)
            or not all(np.isfinite(a).all() for a in (foot_b, hip_b, rotations))):
        raise ValueError('Posture needs nonempty finite [N,4,3] feet/hips and [N,3,3] rotations')
    # Signed left-minus-right lateral distance penalizes crossing as well.
    width = foot_b[:, 0, 1] - foot_b[:, 1, 1]
    hips_w_relative = np.einsum('nij,nkj->nki', rotations, hip_b)
    height_gap = hips_w_relative[:, :2, 2].mean(1) - hips_w_relative[:, 2:, 2].mean(1)
    return dict(front_width_m=distribution(width), front_width_below_0p42_fraction=float(np.mean(width < .42)),
                front_above_rear_hip_m=distribution(height_gap),
                inclination_deg=distribution(np.degrees(np.arccos(np.clip(rotations[:, 2, 2], -1, 1)))),
                abs_pitch_deg=distribution(np.degrees(np.arcsin(np.clip(np.abs(rotations[:, 2, 0]), 0, 1)))))


def review(directory):
    directory = Path(directory).resolve(strict=True)
    manifest = json.loads((directory / 'manifest.json').read_text())
    summary = json.loads((directory / 'summary.json').read_text())
    with gzip.open(directory / 'physics_200hz.jsonl.gz', 'rt') as stream:
        rows = [json.loads(line) for line in stream]
    if not rows:
        raise ValueError('Empty physics log')
    isaac = 'joint_position_rad' in rows[0]
    times = np.array([r['time_s'] for r in rows])
    if not np.allclose(np.diff(times), .005, atol=1e-9, rtol=0):
        raise ValueError('Review requires contiguous 200Hz samples')
    if (len(times) != 6000 or not np.isclose(times[0], .005, atol=1e-9, rtol=0)
            or not np.isclose(times[-1], 30., atol=1e-9, rtol=0)
            or summary.get('duration_s') != 30.):
        raise ValueError('This comparison requires complete 30s first-episode logs, not a short failed trace')
    if not isaac and summary.get('reason') != 'duration_complete':
        raise ValueError('MuJoCo comparison requires a completed first episode')
    if isaac and summary.get('status') != 'completed_sampled_play':
        raise ValueError('Isaac comparison requires a completed first episode')
    spec = importlib.util.spec_from_file_location('vd14_review_fk', ASSETS / 'vd11_geometry_audit.py')
    fk_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fk_module)
    fk = fk_module.GunRobotGeometryAudit(URDF)
    names = manifest['joint_names' if isaac else 'joint_order']
    if manifest['geometry_profile']['urdf_sha256'] != sha256(URDF):
        raise ValueError('Logged geometry is not the same measured mount75 model')
    num_envs = manifest['num_envs'] if isaac else 1
    settings, legacy = Settings(), legacy_metrics()
    results = []
    for env in range(num_envs):
        def array(isaac_key, mj_key=None):
            return np.array([r[isaac_key][env] if isaac else r[mj_key] for r in rows])
        q = array('joint_position_rad', 'joint_position18_rad')
        quat = array('base_quat_wxyz', 'base_quaternion_wxyz')
        base = array('base_position_w', 'base_position_w_m')
        tcp = array('tcp_position_w', 'tcp_position_w_m')
        command = array('tcp_command_world_50hz_zoh', 'tcp_target_w_50hz_zoh_m')
        forces = array('foot_contact_force_w', 'foot_force_world_n')
        velocities = array('foot_velocity_w', 'foot_velocity_world_m_s')
        rot = np.array([rotation_wxyz(v) for v in quat])
        feet_b, hips_b = [], []
        for sample in q:
            transforms = fk._fk(fk._positions(names, sample))
            feet_b.append([transforms[n + '_foot'][1] for n in ('FL', 'FR', 'RL', 'RR')])
            hips_b.append([transforms[n + '_hip'][1] for n in ('FL', 'FR', 'RL', 'RR')])
        feet_b, hips_b = np.array(feet_b), np.array(hips_b)
        predicted_feet_w = base[:, None, :] + np.einsum('nij,nkj->nki', rot, feet_b)
        if 'foot_position_w' in rows[0] or not isaac:
            feet_w = array('foot_position_w', 'foot_position_world_m')
            fk_error = float(np.max(np.linalg.norm(predicted_feet_w - feet_w, axis=-1)))
            if fk_error > .002:
                raise ValueError(f'Foot origin FK/log mismatch: {fk_error}m')
            feet_b = np.einsum('nji,nkj->nki', rot, feet_w-base[:, None, :])
        else:
            fk_error, feet_w = None, predicted_feet_w
        selected = (times >= 3.-1e-9) & (times <= 29.+1e-9)
        motion = filtered_metrics(tcp, command, times, settings, legacy)
        height = feet_w[:, :, 2] - .032
        # MuJoCo native contact is logged from the beginning-state solve,
        # one interval before the row's end-state pose/velocity.
        if isaac:
            impacts = touchdowns(forces, velocities, height, times, times, settings, ('FL','FR','RL','RR'))
        else:
            impacts = touchdowns(forces[1:], velocities[:-1], height[:-1], times[:-1], times[:-1],
                                 settings, ('FL','FR','RL','RR'))
        results.append(dict(env_id=env, posture=posture(feet_b[selected], hips_b[selected], rot[selected]),
                            logged_foot_fk_max_error_m=fk_error, motion=motion,
                            tracking_error_raw_xyz_mm=vector_rms((tcp-command)[selected]),
                            gun_axis_error_deg=distribution(np.degrees(array('gun_axis_error_rad', 'gun_axis_error_rad')[selected])),
                            foot_impacts=impacts))
    return dict(directory=str(directory), checkpoint=manifest['checkpoint'],
                schema='isaac' if isaac else 'mujoco', summary=summary,
                command_source=manifest.get('command_source', manifest.get('command_mode')),
                reset_mode=manifest.get('reset_mode'), seed=manifest.get('seed'),
                measurement_window_s=[3.,29.], native_contact_phase_aligned=not isaac,
                foot_fk_source_sha256=sha256(ASSETS / 'vd11_geometry_audit.py'),
                urdf_sha256=sha256(URDF), trace_sha256=sha256(directory / 'physics_200hz.jsonl.gz'),
                environments=results, deployment_accepted=False,
                note='Within-simulator sampled comparisons only; no convergence, robustness or visual acceptance claim.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('logs', type=Path, nargs='+')
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    reports = [review(path) for path in args.logs]
    args.output_root.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix='posture_review_', dir=args.output_root)) / 'comparison.json'
    with out.open('x') as stream:
        json.dump(dict(reviews=reports), stream, indent=2, allow_nan=False)
    print(out)
    for report in reports:
        for env in report['environments']:
            print(report['directory'], env['env_id'], json.dumps(env['posture']))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
