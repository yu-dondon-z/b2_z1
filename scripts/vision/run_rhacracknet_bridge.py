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
    parser.add_argument("--min-image-v-span-ratio", type=float, default=0.35)
    parser.add_argument("--near-field-v-ratio", type=float, default=0.55)
    parser.add_argument("--image-border-margin-px", type=int, default=24)
    parser.add_argument("--min-mask-area-ratio", type=float, default=0.002)
    parser.add_argument("--max-mask-area-ratio", type=float, default=0.12)
    parser.add_argument("--min-path-straightness", type=float, default=0.65)
    parser.add_argument("--min-forward-cosine", type=float, default=0.65)
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
            overlay = cv2.addWeighted(image, 0.72, mask_bgr, 0.28, 0.0)
            status_text = "no valid 3-D trajectory"
            try:
                trajectory = extract_centerline(
                    segmentation.mask,
                    centerline_cfg,
                    ((image.shape[1] - 1) / 2.0, float(image.shape[0] - 1)),
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
                mask_area_ratio = float(np.mean(segmentation.mask > 0))
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
                        "mask_area_ratio": mask_area_ratio,
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
                    write_json_atomic(exchange / "latest_trajectory.json", payload)
                    write_json_atomic(
                        exchange / "latest_vision_status.json",
                        {
                            "source_sequence": sequence,
                            "valid": True,
                            "valid_streak": valid_streak,
                            "mean_waypoint_confidence": mean_waypoint_confidence,
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
                cv2.putText(mask_bgr, "RHACrackNet mask", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
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
