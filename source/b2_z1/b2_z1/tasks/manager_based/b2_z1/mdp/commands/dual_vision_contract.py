"""Simulator-independent contracts and geometry for the dual-camera controller."""

from __future__ import annotations

import math
import numpy as np


SCHEMA = "b2_z1_dual_vision_v1"


def finite_array(value, shape=None):
    array = np.asarray(value, dtype=np.float64)
    if (shape is not None and array.shape != shape) or not np.isfinite(array).all():
        raise ValueError("nonfinite or incorrectly shaped visual geometry")
    return array


def validate_capture(stream, snapshot, generation, now, max_age_s):
    """Tie a result to an exported capture; publication never refreshes its age."""
    if not isinstance(stream, dict):
        raise ValueError("missing observation")
    sequence = stream.get("sequence")
    if type(sequence) is not int or sequence != snapshot["sequence"]:
        raise ValueError("unknown capture sequence")
    if stream.get("capture_generation") != generation or snapshot["capture_generation"] != generation:
        raise ValueError("observation belongs to a different reset")
    capture = stream.get("capture_timestamp")
    if not isinstance(capture, (float, int)) or not math.isfinite(capture):
        raise ValueError("missing capture clock")
    if abs(capture - snapshot["capture_timestamp"]) > 1.0e-6:
        raise ValueError("capture timestamp differs from exported frame")
    if not -0.002 <= now - capture <= max_age_s:
        raise ValueError("expired or future capture")


def validate_stream(stream, snapshot, generation, now, max_age_s, rear=False):
    validate_capture(stream, snapshot, generation, now, max_age_s)
    if stream.get("valid") is not True:
        raise ValueError(str(stream.get("reason", "invalid observation")))
    points = finite_array(stream.get("points_b"))
    if points.ndim != 2 or points.shape[1] != 3 or not 3 <= len(points) <= 4096:
        raise ValueError("a visual path requires 3..4096 xyz points")
    if np.max(np.abs(points)) > 20 or np.linalg.norm(np.ptp(points, axis=0)) < 0.02:
        raise ValueError("degenerate visual path")
    result = dict(stream, points_b=points)
    if rear:
        result["tcp_measured_b"] = finite_array(stream.get("tcp_measured_b"), (3,))
        plane = finite_array(stream.get("ground_plane_b"), (4,))
        length = np.linalg.norm(plane[:3])
        if not 0.5 <= length <= 2 or abs(plane[2] / length) < 0.4:
            raise ValueError("invalid ground plane")
        result["ground_plane_b"] = plane / length
    return result


def rear_target(points_b, plane_b, tcp_b, up_b, work_x, height_m,
                feedback_gain, max_correction_m):
    """Make a local visual target, including feedback from the observed TCP."""
    points = finite_array(points_b)
    plane = finite_array(plane_b, (4,))
    tcp = finite_array(tcp_b, (3,))
    up = finite_array(up_b, (3,))
    up /= max(np.linalg.norm(up), 1.0e-12)
    denominator = float(plane[:3] @ up)
    if abs(denominator) < 0.4:
        raise ValueError("ground is not observable along gravity")
    order = np.argsort(points[:, 0])
    points = points[order]
    x = 0.5 * (work_x[0] + work_x[1])
    if x < points[0, 0] - 0.12 or x > points[-1, 0] + 0.12:
        raise ValueError("rear seam does not cover the tool workspace")
    y = float(np.interp(x, points[:, 0], points[:, 1]))
    ground = np.array([x, y, -(plane[0] * x + plane[1] * y + plane[3]) / plane[2]])
    desired = ground + height_m * up
    error = desired - tcp
    correction = feedback_gain * error
    norm = np.linalg.norm(correction)
    if norm > max_correction_m:
        correction *= max_correction_m / norm
    projection = tcp - up * ((plane[:3] @ tcp + plane[3]) / denominator)
    starts, deltas = points[:-1], np.diff(points, axis=0)
    fractions = np.sum((projection - starts) * deltas, axis=1) / np.maximum(
        np.sum(deltas * deltas, axis=1), 1.0e-12)
    nearest = starts + np.clip(fractions, 0, 1)[:, None] * deltas
    projection_error = float(np.linalg.norm(nearest - projection, axis=1).min())
    return desired + correction, desired, correction, projection_error
