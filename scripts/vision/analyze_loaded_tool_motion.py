"""Read a playback CSV and report shake without mistaking a stopped robot for success."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def analyze(path, start_s=12.0, end_s=55.0):
    rows = list(csv.DictReader(Path(path).open(newline="")))
    if not rows:
        raise ValueError("empty motion trace")
    active = [r for r in rows if start_s <= float(r['time_s']) <= end_s]
    if len(active) < 10:
        raise ValueError("insufficient steady-window samples")
    if any(int(r['reset']) for r in active):
        raise ValueError("steady window contains resets; analyze separate continuous windows")
    values = lambda key, rs=active: np.array([float(r[key]) for r in rs])
    t = values('time_s')
    dt = np.median(np.diff(t))
    if not np.all(np.isfinite(t)) or dt <= 0 or not np.all(np.diff(t) > 0):
        raise ValueError("invalid trace timestamps")
    stats = {}
    for key in ('base_z', 'actual_z', 'target_z', 'joint2', 'joint3', 'joint4'):
        a = values(key)
        if not np.all(np.isfinite(a)):
            raise ValueError('nonfinite '+key)
        detrended = a - np.polyval(np.polyfit(t, a, 1), t)
        f = np.fft.rfftfreq(len(a), dt)
        band = (f >= .8) & (f <= 5.)
        spectrum = abs(np.fft.rfft(detrended))
        stats[key] = {
            'mean': float(a.mean()),
            'p95_minus_p05': float(np.quantile(a, .95)-np.quantile(a, .05)),
            'rms_detrended': float(np.std(detrended)),
            'dominant_0p8_to_5hz': float(f[band][np.argmax(spectrum[band])])
                if np.any(band) and np.std(detrended) > 1.e-5 else None,
        }
    xy = np.column_stack((values('base_x'), values('base_y')))
    errors = np.column_stack([values('target_'+k)-values('actual_'+k) for k in ('x','y','z')])
    return {
        'trace': str(Path(path).resolve()), 'window_s': [float(t[0]), float(t[-1])],
        'window_travel_m': float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()),
        'whole_trace_resets': sum(int(r['reset']) for r in rows),
        'tcp_xyz_error_p95_m': float(np.quantile(np.linalg.norm(errors, axis=1), .95)),
        'joint4_within_0p01rad_of_lower_limit_fraction': float(np.mean(values('joint4') <= -1.5184364+.01)),
        'motion': stats,
        'limitations': 'P95-P05 is a robust spread, not peak-to-peak; single replay; gun orientation is not recorded in this legacy CSV.'
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('traces', nargs='+', type=Path)
    parser.add_argument('--start-s', type=float, default=12.)
    parser.add_argument('--end-s', type=float, default=55.)
    args = parser.parse_args()
    print(json.dumps([analyze(p, args.start_s, args.end_s) for p in args.traces], indent=2))
