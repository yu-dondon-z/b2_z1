"""Pure, explicit stages shared by configuration and external validation."""
import math

STAGES = (
    {"name": "clear_hold", "height_m": .07, "hose_kg": 0., "moving": False},
    {"name": "one_kg_hold", "height_m": .05, "hose_kg": 1., "moving": False},
    {"name": "three_kg_hold", "height_m": .03, "hose_kg": 3., "moving": False},
    {"name": "weld_height_hold", "height_m": .01, "hose_kg": 3., "moving": False},
    {"name": "weld_height_motion", "height_m": .01, "hose_kg": 3., "moving": True},
)


def stage_spec(index):
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(STAGES):
        raise ValueError("VB13 stage must be an integer from 0 to 4")
    return dict(STAGES[index])


def trace_acceptance(rows, moving=False, settle_s=5.):
    """Strict deterministic validation, not training reward or movement alone."""
    rows = [dict(r) for r in rows]
    if len(rows) < 1500:
        return False, {"reason": "Need at least 30 simulated seconds"}
    if any(int(r['reset']) for r in rows):
        return False, {"reason": "Physical/timeout reset in validation", "resets": sum(int(r['reset']) for r in rows)}
    ready = [r for r in rows if float(r['time_s']) >= settle_s]
    required = ['time_s', 'body_tilt_deg', 'base_z', 'gun_contact_force_n',
                'gun_bottom_clearance_m', 'command_vx_b', 'base_x', 'path_lateral_error_b', 'path_heading_error_b']
    required += [s+a for s in ['actual_', 'target_'] for a in 'xyz']
    if any(not math.isfinite(float(r[k])) for r in rows for k in required):
        return False, {'reason': 'Nonfinite trace values'}
    times = [float(r['time_s']) for r in rows]
    if times[-1]-times[0] < 29.9 or any(b <= a for a,b in zip(times, times[1:])) or not ready:
        return False, {'reason': 'Insufficient or invalid simulation timestamps'}
    arm_names = [f'joint{i}' for i in range(1,7)]
    for i, r in enumerate(rows):
        if not all(math.isfinite(float(r[k])) for k in arm_names):
            return False, {'reason': 'Nonfinite measured arm state'}
        speed = 0. if i == 0 else max(abs(float(r[k])-float(rows[i-1][k]))/(times[i]-times[i-1]) for k in arm_names)
        # Work on detached row dictionaries, never alter the recorded CSV.
        r = dict(r)
        r['_arm_speed'] = speed
        rows[i] = r
    ready = [r for r in rows if float(r['time_s']) >= settle_s]
    def measured(r):
        e = [(float(r['actual_'+a])-float(r['target_'+a])) for a in 'xyz']
        vals = e + [float(r[k]) for k in ['body_tilt_deg','gun_contact_force_n','gun_bottom_clearance_m','command_vx_b']]
        finite = all(math.isfinite(v) for v in vals)
        return finite and (sum(v*v for v in e)**.5 < .08) and abs(e[2]) < .01 and (
            float(r['body_tilt_deg']) < math.degrees(.10)) and (
            float(r['gun_contact_force_n']) < 5.) and float(r['gun_bottom_clearance_m']) > .005 and float(r['base_z']) > .45 and r['_arm_speed'] < .8
    held = sum(measured(r) for r in ready)/max(len(ready), 1)
    nonzero = sum(float(r['command_vx_b'])>.05 for r in ready)/max(len(ready), 1)
    dx = float(rows[-1]['base_x'])-float(rows[0]['base_x'])
    ordered_lateral = sorted(abs(float(r['path_lateral_error_b'])) for r in ready)
    ordered_heading = sorted(abs(float(r['path_heading_error_b'])) for r in ready)
    lateral95 = ordered_lateral[min(math.ceil(.95*len(ready))-1, len(ready)-1)]
    heading95 = ordered_heading[min(math.ceil(.95*len(ready))-1, len(ready)-1)]
    accepted = held >= .90 and (not moving or (nonzero >= .70 and dx > 3. and lateral95 < .10 and heading95 < .12))
    return accepted, {"ready_fraction_after_5s": held, "command_fraction_after_5s": nonzero,
                      "net_world_x_m": dx, 'path_lateral_p95_m': lateral95, 'path_heading_p95_rad': heading95,
                      "reason": "pass" if accepted else "TCP/contact/motion thresholds not met"}
