#!/usr/bin/env python3
"""Convert the simulated D435i RGB-D ring buffer into B2-frame crack paths."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
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
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--spacing-px", type=float, default=8.0)
    parser.add_argument("--spacing-m", type=float, default=0.03)
    parser.add_argument("--max-3d-step-m", type=float, default=0.25)
    parser.add_argument("--min-forward-span-m", type=float, default=0.15)
    parser.add_argument("--min-path-length-m", type=float, default=0.20)
    parser.add_argument("--max-frames", type=int, default=0, help="0 runs until Ctrl-C")
    parser.add_argument("--poll-period", type=float, default=0.01)
    parser.add_argument("--save-overlay", action="store_true")
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
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


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
    segmenter = CrackSegmenter(args.checkpoint, args.model, args.device, args.threshold)
    centerline_cfg = CenterlineConfig(waypoint_spacing_px=args.spacing_px)
    last_sequence = -1
    processed = 0
    warmed_up = False
    print(f"Watching RGB-D bridge: {metadata_path}")

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
        try:
            trajectory = extract_centerline(
                segmentation.mask,
                centerline_cfg,
                ((image.shape[1] - 1) / 2.0, float(image.shape[0] - 1)),
            )
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
            payload = {
                "timestamp": datetime.now().astimezone().isoformat(),
                "source_sequence": sequence,
                "frame_id": "b2_base",
                "camera_name": metadata.get("camera_name", "d435i"),
                "inference_ms": float(segmentation.inference_ms),
                "valid_3d_ratio": float(np.mean(valid)) if len(valid) else 0.0,
                "path_length_m": path_length_m,
                "forward_span_m": forward_span_m,
                "waypoints_uv": trajectory.waypoints_uv.tolist(),
                "waypoint_depth_m": [
                    float(value) if np.isfinite(value) else None for value in depths_m
                ],
                "control_waypoints_robot_xyz": control.tolist(),
            }
            write_json_atomic(exchange / "latest_trajectory.json", payload)
            print(
                f"frame={sequence} inference={segmentation.inference_ms:.1f} ms "
                f"valid={payload['valid_3d_ratio']:.2f} control_points={len(control)}"
            )
            if args.save_overlay:
                overlay = draw_trajectory_overlay(image, trajectory)
                overlay_tmp = exchange / ".latest_overlay.tmp.png"
                cv2.imwrite(str(overlay_tmp), overlay)
                overlay_tmp.replace(exchange / "latest_overlay.png")
        except (KeyError, RuntimeError, TypeError, ValueError) as error:
            # Remove any previous result so a newly started controller cannot
            # briefly consume an old path. Missing/stale input is a zero-speed
            # condition in the WBC command term.
            (exchange / "latest_trajectory.json").unlink(missing_ok=True)
            write_json_atomic(
                exchange / "latest_vision_status.json",
                {"source_sequence": sequence, "valid": False, "reason": str(error)},
            )
            print(f"frame={sequence} no valid trajectory: {error}")


if __name__ == "__main__":
    main()
