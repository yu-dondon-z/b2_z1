#!/usr/bin/env python3
"""Measure a visual-play trace against the actual seamless-road crack mesh.

Before VB6, file-mode path_lateral_error_b was a forward lookahead coordinate.
VB6 instead estimates local cross-track error from camera/cache points. This
independent geometry check does not depend on either controller definition.
Defaults match SeamlessCrackedRoadHalfCfg
for a single environment at world origin; record overrides for other scenes.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def distances_to_polyline(points, vertices):
    starts = vertices[:-1]
    segments = vertices[1:] - starts
    length_sq = np.sum(segments * segments, axis=1)
    distances = []
    for first in range(0, len(points), 256):
        offset = points[first:first + 256, None, :] - starts[None, :, :]
        fraction = np.clip(np.sum(offset * segments[None, :, :], axis=2) / length_sq, 0, 1)
        residual = offset - fraction[:, :, None] * segments[None, :, :]
        distances.extend(np.sqrt(np.min(np.sum(residual * residual, axis=2), axis=1)))
    return np.asarray(distances)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--amplitude", type=float, default=0.40)
    parser.add_argument("--wavelength", type=float, default=20.0)
    parser.add_argument("--curve-half-length", type=float, default=20.0)
    parser.add_argument("--road-half-length", type=float, default=95.0)
    parser.add_argument("--mesh-step", type=float, default=0.05)
    args = parser.parse_args()
    parameters = (args.amplitude, args.wavelength, args.curve_half_length,
                  args.road_half_length, args.mesh_step)
    if (not np.isfinite(parameters).all() or args.wavelength <= 0 or args.mesh_step <= 0
            or not 0 < args.curve_half_length < args.road_half_length):
        parser.error("Need finite geometry, positive wavelength/step and 0 < curve length < road length")
    with args.trace.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        parser.error("Trace is empty")
    curve_x = np.linspace(-args.curve_half_length, args.curve_half_length,
                          round(2 * args.curve_half_length / args.mesh_step) + 1)
    x = np.concatenate(([-args.road_half_length], curve_x, [args.road_half_length]))
    y = np.zeros_like(x)
    y[1:-1] = args.amplitude * np.sin(2 * np.pi * curve_x / args.wavelength) ** 3
    vertices = np.column_stack((x, y))
    base = np.asarray([[float(row["base_x"]), float(row["base_y"])] for row in rows])
    tcp = np.asarray([[float(row["actual_x"]), float(row["actual_y"])] for row in rows])
    height = np.asarray([float(row["actual_z"]) for row in rows])
    if not all(np.isfinite(values).all() for values in (base, tcp, height)):
        parser.error("Trace contains non-finite poses")
    base_error = distances_to_polyline(base, vertices)
    tcp_error = distances_to_polyline(tcp, vertices)
    print(json.dumps({
        "trace": str(args.trace.resolve()),
        "geometry": {key: getattr(args, key) for key in (
            "amplitude", "wavelength", "curve_half_length", "road_half_length", "mesh_step")},
        "steps": len(rows),
        "base_true_crack_distance_p95_m": float(np.quantile(base_error, 0.95)),
        "base_true_crack_distance_max_m": float(base_error.max()),
        "tcp_true_crack_planar_distance_p95_m": float(np.quantile(tcp_error, 0.95)),
        "tcp_height_above_road_p05_m": float(np.quantile(height, 0.05)),
        "tcp_height_above_road_p95_m": float(np.quantile(height, 0.95)),
    }, indent=2))


if __name__ == "__main__":
    main()
