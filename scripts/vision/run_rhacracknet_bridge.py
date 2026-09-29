#!/usr/bin/env python3
"""Convert the simulated D435i RGB-D ring buffer into B2-frame crack paths."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exchange-dir", default="/tmp/b2_z1_vision_bridge")
    parser.add_argument("--rhacracknet-root", default="/home/dong/RHACrackNet/code")
    parser.add_argument(
        "--checkpoint",
        default=(
            "/home/dong/RHACrackNet/code/exps/"
            "rhanet_dep_CamCrack789_b2_stage1/ckpt/best_ckpt.pth"
        ),
    )
    parser.add_argument("--model", choices=("RHANet", "RHANet_dep"), default="RHANet_dep")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--spacing-px", type=float, default=8.0)
    parser.add_argument("--spacing-m", type=float, default=0.03)
    parser.add_argument("--max-3d-step-m", type=float, default=0.25)
    parser.add_argument("--min-forward-span-m", type=float, default=0.15)
    parser.add_argument("--min-path-length-m", type=float, default=0.20)
    parser.add_argument("--min-waypoint-confidence", type=float, default=0.90)
    parser.add_argument("--min-valid-depth-ratio", type=float, default=0.75)
    # The fixed B2 camera has unavoidable chassis/arm occlusion in the lower
    # image. A 0.20 vertical-span threshold still rejects short blobs while
    # allowing the visible road segment above the occlusion to be tracked.
    parser.add_argument("--min-image-v-span-ratio", type=float, default=0.20)
    parser.add_argument("--near-field-v-ratio", type=float, default=0.55)
    parser.add_argument(
        "--centerline-anchor-v-ratio",
        type=float,
        default=0.58,
        help=(
            "vertical anchor used to choose the crack component; keep it above "
            "the B2/arm self-occlusion visible at the bottom of the built-in camera"
        ),
    )
    parser.add_argument("--image-border-margin-px", type=int, default=24)
    parser.add_argument("--min-mask-area-ratio", type=float, default=0.002)
    parser.add_argument("--max-mask-area-ratio", type=float, default=0.12)
    shadow_filter_group = parser.add_mutually_exclusive_group()
    shadow_filter_group.add_argument(
        "--depth-shadow-filter",
        action="store_true",
        default=False,
        help=(
            "experimental depth-based mask filtering; disabled by default to preserve "
            "the original policy playback path (simulation uses diffuse lighting)"
        ),
    )
    shadow_filter_group.add_argument(
        "--no-depth-shadow-filter",
        dest="depth_shadow_filter",
        action="store_false",
        help="use the original centerline and mask-area checks (default)",
    )
    parser.add_argument(
        "--depth-support-offset-px",
        type=float,
        default=6.0,
        help="normal offset used to compare a candidate centerline with the road on both sides",
    )
    parser.add_argument(
        "--min-crack-depth-contrast-m",
        type=float,
        default=0.015,
        help="minimum center-versus-road depth increase that corroborates a crack waypoint",
    )
    parser.add_argument(
        "--min-depth-supported-waypoint-ratio",
        type=float,
        default=0.60,
        help="fraction of centerline waypoints that must have crack-like RGB-D geometry",
    )
    parser.add_argument(
        "--min-depth-probe-ratio",
        type=float,
        default=0.75,
        help="fraction of centerline waypoints requiring valid center and two-sided depth probes",
    )
    parser.add_argument(
        "--depth-support-dilation-px",
        type=int,
        default=5,
        help="dilation used to reconnect depth-supported crack pixels before thinning",
    )
    parser.add_argument("--min-path-straightness", type=float, default=0.65)
    # Curved paths can have a large endpoint bearing even when their local
    # tangent remains valid. 0.35 accepts bends up to about 69 degrees while
    # still rejecting reversed or mostly lateral segmentation artifacts.
    parser.add_argument("--min-forward-cosine", type=float, default=0.35)
    parser.add_argument("--min-consecutive-valid", type=int, default=3)
    parser.add_argument("--max-heading-jump-deg", type=float, default=20.0)
    parser.add_argument("--max-near-lateral-jump-m", type=float, default=0.20)
    parser.add_argument("--max-frames", type=int, default=0, help="0 runs until Ctrl-C")
    parser.add_argument("--poll-period", type=float, default=0.01)
    parser.add_argument("--log-period-frames", type=int, default=15)
    parser.add_argument("--save-overlay", action="store_true")
    parser.add_argument(
        "--display",
        action="store_true",
        help="show RGB, segmentation mask, and extracted centerline in a live OpenCV window",
    )
    return parser.parse_args()


def quaternion_wxyz_to_matrix(quaternion) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    norm = float(np.linalg.norm((w, x, y, z)))
    if norm <= 1.0e-12:
        raise ValueError("camera quaternion cannot be zero")
    w, x, y, z = (w / norm, x / norm, y / norm, z / norm)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def write_json_atomic(path: Path, payload: dict) -> None:
    # PID-scoped temporary files keep two accidentally launched bridge
    # processes from stealing each other's atomic-replace source.
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _median_depth_m(
    depth: np.ndarray,
    u: float,
    v: float,
    scale_m_per_unit: float,
    radius_px: int = 1,
) -> float:
    """Return a small-patch median, or NaN when the probe leaves the image."""
    center_u, center_v = int(round(u)), int(round(v))
    height, width = depth.shape
    if (
        center_u - radius_px < 0
        or center_u + radius_px >= width
        or center_v - radius_px < 0
        or center_v + radius_px >= height
    ):
        return float("nan")
    patch = depth[
        center_v - radius_px : center_v + radius_px + 1,
        center_u - radius_px : center_u + radius_px + 1,
    ].astype(np.float64, copy=False)
    values_m = patch.reshape(-1) * scale_m_per_unit
    values_m = values_m[np.isfinite(values_m) & (values_m > 0.0)]
    if len(values_m) < 3:
        return float("nan")
    return float(np.median(values_m))


def centerline_depth_support(
    waypoints_uv: np.ndarray,
    depth: np.ndarray,
    scale_m_per_unit: float,
    normal_offset_px: float,
    min_depth_contrast_m: float,
) -> tuple[float, float, np.ndarray]:
    """Measure whether a dark 2-D line is also a physical depth depression.

    Shadows change RGB intensity but leave the road depth continuous. A real
    trench is farther from the camera than road samples taken on both sides of
    its local centerline. The symmetric comparison cancels the smooth depth
    gradient caused by the camera looking obliquely at the ground.
    """
    waypoints = np.asarray(waypoints_uv, dtype=np.float64).reshape(-1, 2)
    contrasts_m = np.full(len(waypoints), np.nan, dtype=np.float64)
    for index, waypoint in enumerate(waypoints):
        before = waypoints[max(0, index - 1)]
        after = waypoints[min(len(waypoints) - 1, index + 1)]
        tangent = after - before
        tangent_norm = float(np.linalg.norm(tangent))
        if tangent_norm <= 1.0e-6:
            continue
        normal = np.array((-tangent[1], tangent[0]), dtype=np.float64) / tangent_norm
        center_depth = _median_depth_m(
            depth, waypoint[0], waypoint[1], scale_m_per_unit
        )
        side_depths = [
            _median_depth_m(
                depth,
                waypoint[0] + sign * normal_offset_px * normal[0],
                waypoint[1] + sign * normal_offset_px * normal[1],
                scale_m_per_unit,
            )
            for sign in (-1.0, 1.0)
        ]
        if np.isfinite(center_depth) and np.all(np.isfinite(side_depths)):
            contrasts_m[index] = center_depth - 0.5 * sum(side_depths)

    usable = np.isfinite(contrasts_m)
    valid_probe_ratio = float(np.mean(usable)) if len(usable) else 0.0
    support_ratio = (
        float(np.mean(contrasts_m[usable] >= min_depth_contrast_m))
        if np.any(usable)
        else 0.0
    )
    return support_ratio, valid_probe_ratio, contrasts_m


def depth_depression_support_mask(
    depth: np.ndarray,
    scale_m_per_unit: float,
    offset_px: float,
    min_depth_contrast_m: float,
    dilation_px: int,
) -> np.ndarray:
    """Return pixels that are deeper than their surroundings in any direction."""
    if depth is None or depth.ndim != 2:
        raise ValueError("depth must be a non-empty HxW array")
    offset = max(1, int(round(offset_px)))
    diagonal = max(1, int(round(offset / np.sqrt(2.0))))
    depth_m = depth.astype(np.float64, copy=False) * scale_m_per_unit
    height, width = depth_m.shape
    padded = np.pad(depth_m, offset, mode="constant", constant_values=np.nan)
    supported = np.zeros((height, width), dtype=bool)
    for delta_u, delta_v in (
        (offset, 0),
        (0, offset),
        (diagonal, diagonal),
        (diagonal, -diagonal),
    ):
        side_a = padded[
            offset + delta_v : offset + delta_v + height,
            offset + delta_u : offset + delta_u + width,
        ]
        side_b = padded[
            offset - delta_v : offset - delta_v + height,
            offset - delta_u : offset - delta_u + width,
        ]
        contrast = depth_m - 0.5 * (side_a + side_b)
        supported |= (
            np.isfinite(contrast)
            & (depth_m > 0.0)
            & (side_a > 0.0)
            & (side_b > 0.0)
            & (contrast >= min_depth_contrast_m)
        )

    kernel_size = max(1, int(dilation_px))
    if kernel_size > 1:
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        supported = cv2.dilate(supported.astype(np.uint8), kernel) > 0
    return supported


def extract_depth_supported_centerline(
    mask: np.ndarray,
    depth: np.ndarray,
    centerline_cfg,
    anchor_uv: tuple[float, float],
    extract_centerline_fn,
    scale_m_per_unit: float,
    normal_offset_px: float,
    min_depth_contrast_m: float,
    min_support_ratio: float,
    min_probe_ratio: float,
    support_dilation_px: int,
):
    """Choose a segmented component using RGB-D evidence, not RGB proximity alone."""
    depth_supported = depth_depression_support_mask(
        depth,
        scale_m_per_unit,
        normal_offset_px,
        min_depth_contrast_m,
        support_dilation_px,
    )
    filtered_mask = ((mask > 0) & depth_supported).astype(np.uint8) * 255
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (filtered_mask > 0).astype(np.uint8), connectivity=8
    )
    candidates = []
    rejected = []
    for label in range(1, component_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < centerline_cfg.min_component_area:
            continue
        component_mask = np.zeros_like(mask, dtype=np.uint8)
        component_mask[labels == label] = 255
        try:
            trajectory = extract_centerline_fn(component_mask, centerline_cfg, anchor_uv)
        except (RuntimeError, TypeError, ValueError):
            continue
        support_ratio, probe_ratio, contrasts_m = centerline_depth_support(
            trajectory.waypoints_uv,
            depth,
            scale_m_per_unit,
            normal_offset_px,
            min_depth_contrast_m,
        )
        anchor_distance_px = float(
            np.linalg.norm(trajectory.waypoints_uv - np.asarray(anchor_uv), axis=1).min()
        )
        image_v_span_px = float(np.ptp(trajectory.waypoints_uv[:, 1]))
        candidate = (
            support_ratio,
            probe_ratio,
            image_v_span_px,
            trajectory.path_length_px,
            -anchor_distance_px,
            trajectory,
            contrasts_m,
        )
        if support_ratio >= min_support_ratio and probe_ratio >= min_probe_ratio:
            candidates.append(candidate)
        else:
            rejected.append(candidate)

    if not candidates:
        if rejected:
            best = max(rejected, key=lambda item: item[:5])
            raise RuntimeError(
                "no component has crack-like RGB-D depth support "
                f"(best support={best[0]:.2f}, probes={best[1]:.2f})"
            )
        raise RuntimeError("no usable segmentation component was found")

    selected = max(candidates, key=lambda item: item[:5])
    diagnostics = {
        "depth_support_ratio": selected[0],
        "depth_probe_ratio": selected[1],
        "mean_depth_contrast_m": float(np.nanmean(selected[6])),
        "depth_supported_mask_area_ratio": float(np.mean(filtered_mask > 0)),
        "segmentation_component_count": len(candidates) + len(rejected),
    }
    return selected[5], diagnostics


def main() -> None:
    args = parse_args()
    code_root = Path(args.rhacracknet_root).expanduser().resolve()
    if str(code_root) not in sys.path:
        sys.path.insert(0, str(code_root))

    from crack_inference import CrackSegmenter
    from crack_trajectory import (
        CenterlineConfig,
        DepthProjectionConfig,
        PinholeCamera,
        build_metric_trajectory,
        draw_trajectory_overlay,
        extract_centerline,
        project_pixels_with_depth,
    )

    exchange = Path(args.exchange_dir).expanduser().resolve()
    exchange.mkdir(parents=True, exist_ok=True)
    metadata_path = exchange / "latest_frame.json"
    (exchange / "latest_trajectory.json").unlink(missing_ok=True)
    segmenter = CrackSegmenter(args.checkpoint, args.model, args.device, args.threshold)
    centerline_cfg = CenterlineConfig(waypoint_spacing_px=args.spacing_px)
    last_sequence = -1
    processed = 0
    warmed_up = False
    valid_streak = 0
    previous_heading = None
    previous_near_lateral = None
    print(f"Watching RGB-D bridge: {metadata_path}")

    try:
        while args.max_frames <= 0 or processed < args.max_frames:
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                sequence = int(metadata["sequence"])
            except (FileNotFoundError, OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                time.sleep(args.poll_period)
                continue
            if sequence == last_sequence:
                time.sleep(args.poll_period)
                continue
            if sequence < last_sequence:
                # Isaac restarts its camera sequence at zero. Do not count a
                # frame from the previous simulation toward temporal gating.
                valid_streak = 0
                previous_heading = None
                previous_near_lateral = None

            image = cv2.imread(str(exchange / metadata["rgb_file"]), cv2.IMREAD_COLOR)
            depth = cv2.imread(str(exchange / metadata["depth_file"]), cv2.IMREAD_UNCHANGED)
            if image is None or depth is None or depth.ndim != 2 or depth.shape != image.shape[:2]:
                time.sleep(args.poll_period)
                continue
            last_sequence = sequence
            processed += 1

            if not warmed_up:
                segmenter.warmup(image.shape[0], image.shape[1])
                warmed_up = True
            segmentation = segmenter.predict_bgr(image)
            mask_bgr = cv2.cvtColor(segmentation.mask, cv2.COLOR_GRAY2BGR)
            mask_label = "RHACrackNet mask"
            overlay = cv2.addWeighted(image, 0.72, mask_bgr, 0.28, 0.0)
            status_text = "no valid 3-D trajectory"
            try:
                anchor_uv = (
                    (image.shape[1] - 1) / 2.0,
                    args.centerline_anchor_v_ratio * (image.shape[0] - 1),
                )
                depth_diagnostics = {"depth_shadow_filter_enabled": args.depth_shadow_filter}
                if args.depth_shadow_filter:
                    trajectory, support_diagnostics = extract_depth_supported_centerline(
                        segmentation.mask,
                        depth,
                        centerline_cfg,
                        anchor_uv,
                        extract_centerline,
                        float(metadata.get("depth_unit_m", 0.001)),
                        args.depth_support_offset_px,
                        args.min_crack_depth_contrast_m,
                        args.min_depth_supported_waypoint_ratio,
                        args.min_depth_probe_ratio,
                        args.depth_support_dilation_px,
                    )
                    depth_diagnostics.update(support_diagnostics)
                    mask_bgr = cv2.cvtColor(trajectory.cleaned_mask, cv2.COLOR_GRAY2BGR)
                    mask_label = "RGB-D filtered mask"
                else:
                    trajectory = extract_centerline(
                        segmentation.mask, centerline_cfg, anchor_uv
                    )
                height, width = image.shape[:2]
                waypoints_uv = trajectory.waypoints_uv
                waypoint_u = np.clip(np.rint(waypoints_uv[:, 0]).astype(int), 0, width - 1)
                waypoint_v = np.clip(np.rint(waypoints_uv[:, 1]).astype(int), 0, height - 1)
                waypoint_confidence = segmentation.probability[waypoint_v, waypoint_u]
                mean_waypoint_confidence = float(np.mean(waypoint_confidence))
                if mean_waypoint_confidence < args.min_waypoint_confidence:
                    raise RuntimeError(
                        "centerline confidence is too low "
                        f"({mean_waypoint_confidence:.3f} < {args.min_waypoint_confidence:.3f})"
                    )
                raw_mask_area_ratio = float(np.mean(segmentation.mask > 0))
                mask_area_ratio = (
                    float(np.mean(trajectory.cleaned_mask > 0))
                    if args.depth_shadow_filter
                    else raw_mask_area_ratio
                )
                if not args.min_mask_area_ratio <= mask_area_ratio <= args.max_mask_area_ratio:
                    raise RuntimeError(
                        "segmented area ratio is implausible "
                        f"({mask_area_ratio:.4f}, expected "
                        f"{args.min_mask_area_ratio:.4f}..{args.max_mask_area_ratio:.4f})"
                    )
                image_v_span_ratio = float(np.ptp(waypoints_uv[:, 1]) / height)
                if image_v_span_ratio < args.min_image_v_span_ratio:
                    raise RuntimeError(
                        "centerline does not span enough of the image "
                        f"({image_v_span_ratio:.3f} < {args.min_image_v_span_ratio:.3f})"
                    )
                near_index_uv = int(np.argmax(waypoints_uv[:, 1]))
                near_u, near_v = waypoints_uv[near_index_uv]
                if near_v < args.near_field_v_ratio * height:
                    raise RuntimeError("centerline does not reach the near-field image region")
                if not args.image_border_margin_px <= near_u <= width - 1 - args.image_border_margin_px:
                    raise RuntimeError("near-field centerline touches an image side border")
                intrinsic = np.asarray(metadata["intrinsic_matrix"], dtype=np.float64)
                camera = PinholeCamera(
                    fx=float(intrinsic[0, 0]),
                    fy=float(intrinsic[1, 1]),
                    cx=float(intrinsic[0, 2]),
                    cy=float(intrinsic[1, 2]),
                )
                transform = np.eye(4, dtype=np.float64)
                transform[:3, :3] = quaternion_wxyz_to_matrix(
                    metadata["camera_quaternion_b_ros_wxyz"]
                )
                transform[:3, 3] = np.asarray(metadata["camera_position_b"], dtype=np.float64)
                depth_cfg = DepthProjectionConfig(
                    scale_m_per_unit=float(metadata.get("depth_unit_m", 0.001)),
                    min_depth_m=0.07,
                    max_depth_m=6.0,
                    sample_radius_px=2,
                    min_valid_samples=3,
                )
                points, valid, depths_m = project_pixels_with_depth(
                    trajectory.waypoints_uv, depth, camera, transform, depth_cfg
                )
                control = build_metric_trajectory(
                    points, valid, args.spacing_m, args.max_3d_step_m
                )
                if len(control) < 3:
                    raise RuntimeError("fewer than three valid 3-D control waypoints")
                valid_depth_ratio = float(np.mean(valid)) if len(valid) else 0.0
                if valid_depth_ratio < args.min_valid_depth_ratio:
                    raise RuntimeError(
                        f"valid depth ratio is too low ({valid_depth_ratio:.2f} < "
                        f"{args.min_valid_depth_ratio:.2f})"
                    )
                path_length_m = float(np.linalg.norm(np.diff(control, axis=0), axis=1).sum())
                forward_span_m = float(abs(control[-1, 0] - control[0, 0]))
                if path_length_m < args.min_path_length_m:
                    raise RuntimeError(
                        f"3-D path is too short ({path_length_m:.3f} m < {args.min_path_length_m:.3f} m)"
                    )
                if forward_span_m < args.min_forward_span_m:
                    raise RuntimeError(
                        "3-D path lacks forward extent "
                        f"({forward_span_m:.3f} m < {args.min_forward_span_m:.3f} m)"
                    )
                endpoint_xy = control[-1, :2] - control[0, :2]
                if endpoint_xy[0] < 0.0:
                    endpoint_xy = -endpoint_xy
                endpoint_distance_xy = float(np.linalg.norm(endpoint_xy))
                forward_cosine = float(endpoint_xy[0] / max(endpoint_distance_xy, 1.0e-9))
                if forward_cosine < args.min_forward_cosine:
                    raise RuntimeError(
                        f"path direction is insufficiently forward ({forward_cosine:.3f} < "
                        f"{args.min_forward_cosine:.3f})"
                    )
                straightness = endpoint_distance_xy / max(path_length_m, 1.0e-9)
                if straightness < args.min_path_straightness:
                    raise RuntimeError(
                        f"path is too fragmented/curved ({straightness:.3f} < "
                        f"{args.min_path_straightness:.3f})"
                    )

                heading = float(np.arctan2(endpoint_xy[1], endpoint_xy[0]))
                near_control_index = int(np.argmin(np.abs(control[:, 0] - 0.75)))
                near_lateral = float(control[near_control_index, 1])
                temporally_consistent = True
                if previous_heading is not None:
                    heading_jump = abs(
                        float(np.arctan2(np.sin(heading - previous_heading), np.cos(heading - previous_heading)))
                    )
                    lateral_jump = abs(near_lateral - previous_near_lateral)
                    temporally_consistent = (
                        heading_jump <= np.deg2rad(args.max_heading_jump_deg)
                        and lateral_jump <= args.max_near_lateral_jump_m
                    )
                valid_streak = valid_streak + 1 if temporally_consistent else 1
                previous_heading = heading
                previous_near_lateral = near_lateral

                overlay = draw_trajectory_overlay(image, trajectory)
                if valid_streak < args.min_consecutive_valid:
                    (exchange / "latest_trajectory.json").unlink(missing_ok=True)
                    status_text = f"PENDING: valid frame {valid_streak}/{args.min_consecutive_valid}"
                    write_json_atomic(
                        exchange / "latest_vision_status.json",
                        {
                            "source_sequence": sequence,
                            "valid": False,
                            "pending": True,
                            "valid_streak": valid_streak,
                            "reason": "waiting for temporally consistent frames",
                        },
                    )
                    print(f"frame={sequence} {status_text}")
                else:
                    payload = {
                        "timestamp": datetime.now().astimezone().isoformat(),
                        "source_sequence": sequence,
                        "frame_id": "b2_base",
                        "camera_name": metadata.get("camera_name", "d435i"),
                        "inference_ms": float(segmentation.inference_ms),
                        "mean_waypoint_confidence": mean_waypoint_confidence,
                        "raw_mask_area_ratio": raw_mask_area_ratio,
                        "mask_area_ratio": mask_area_ratio,
                        **depth_diagnostics,
                        "valid_3d_ratio": valid_depth_ratio,
                        "path_length_m": path_length_m,
                        "forward_span_m": forward_span_m,
                        "forward_cosine": forward_cosine,
                        "straightness": straightness,
                        "valid_streak": valid_streak,
                        "waypoints_uv": trajectory.waypoints_uv.tolist(),
                        "waypoint_depth_m": [
                            float(value) if np.isfinite(value) else None for value in depths_m
                        ],
                        "control_waypoints_robot_xyz": control.tolist(),
                    }
                    for key in ("capture_base_position_w", "capture_base_quaternion_wxyz", "capture_generation"):
                        if key in metadata:
                            payload[key] = metadata[key]
                    write_json_atomic(exchange / "latest_trajectory.json", payload)
                    write_json_atomic(
                        exchange / "latest_vision_status.json",
                        {
                            "source_sequence": sequence,
                            "valid": True,
                            "valid_streak": valid_streak,
                            "mean_waypoint_confidence": mean_waypoint_confidence,
                            **depth_diagnostics,
                            "forward_cosine": forward_cosine,
                        },
                    )
                    status_text = (
                        f"VALID: {len(control)} pts, conf {mean_waypoint_confidence:.2f}, "
                        f"span {forward_span_m:.2f} m"
                    )
                    if sequence % max(1, args.log_period_frames) == 0:
                        print(
                            f"frame={sequence} inference={segmentation.inference_ms:.1f} ms "
                            f"confidence={mean_waypoint_confidence:.3f} "
                            f"depth={valid_depth_ratio:.2f} control_points={len(control)}"
                        )
            except (KeyError, RuntimeError, TypeError, ValueError) as error:
                # Remove any previous result so a newly started controller cannot
                # briefly consume an old path. Missing/stale input is a zero-speed
                # condition in the WBC command term.
                (exchange / "latest_trajectory.json").unlink(missing_ok=True)
                valid_streak = 0
                previous_heading = None
                previous_near_lateral = None
                write_json_atomic(
                    exchange / "latest_vision_status.json",
                    {"source_sequence": sequence, "valid": False, "reason": str(error)},
                )
                status_text = f"INVALID: {error}"
                if sequence % max(1, args.log_period_frames) == 0:
                    print(f"frame={sequence} no valid trajectory: {error}")

            if status_text.startswith("VALID"):
                status_color = (40, 220, 40)
            elif status_text.startswith("PENDING"):
                status_color = (0, 210, 255)
            else:
                status_color = (20, 20, 240)
            cv2.putText(
                overlay,
                status_text,
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                status_color,
                2,
                cv2.LINE_AA,
            )
            if args.save_overlay:
                overlay_tmp = exchange / f".latest_overlay.{os.getpid()}.{sequence}.tmp.png"
                if cv2.imwrite(str(overlay_tmp), overlay):
                    overlay_tmp.replace(exchange / "latest_overlay.png")
            if args.display:
                cv2.putText(image, "D435i RGB", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(
                    mask_bgr,
                    mask_label,
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )
                panel = np.hstack((image, mask_bgr, overlay))
                panel = cv2.resize(panel, (1440, 360), interpolation=cv2.INTER_AREA)
                cv2.imshow("B2 crack vision: RGB | mask | 3-D validated centerline", panel)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
    except KeyboardInterrupt:
        print("Vision bridge stopped.")
    finally:
        if args.display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
