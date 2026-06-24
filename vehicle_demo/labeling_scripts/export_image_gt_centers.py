import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent if _SCRIPT_DIR.name == "labeling_scripts" else _SCRIPT_DIR.parents[1]
for _path in (_SCRIPT_DIR, _ROOT / "ultralytics" / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from demo_fuse_radar_camera import (
    VEHICLE_CLASSES,
    default_calibration,
    find_nearest_frame,
    load_calibration_override,
    load_radar_frames,
    match_points_to_boxes,
    parse_axis_map,
    parse_image_timestamp,
    project_points,
    scale_intrinsics,
)
from export_vehicle_centers_mast3r import (
    build_mast3r_predictor,
    choose_neighbor_idx,
    pick_device,
    project_cam_point_to_uv,
)


@dataclass
class RadarFrame:
    timestamp_sec: float
    points_xyz: np.ndarray
    speed_mps: Optional[np.ndarray] = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export image-dominant pseudo ground-truth vehicle centers with radar used only as validation."
    )
    p.add_argument("--images-dir", type=Path, required=True)
    p.add_argument("--radar-csv", type=Path, default=None)
    p.add_argument(
        "--visual-only",
        action="store_true",
        help="Do not require or use radar. YOLO + MASt3R produce the center estimate for every readable image.",
    )
    p.add_argument("--calib-json", type=Path, default=None)
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--pattern", type=str, default="*.jpg")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--model", type=str, default="yolov8m.pt")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--max-dt", type=float, default=0.2)
    p.add_argument("--time-offset", type=float, default=0.0)
    p.add_argument("--auto-offset", action="store_true")
    p.add_argument("--k-scale", type=str, default="auto", choices=["auto", "manual", "none"])
    p.add_argument("--calib-width", type=int, default=None)
    p.add_argument("--calib-height", type=int, default=None)
    p.add_argument("--radar-axis", type=str, default="-y,z,x")
    p.add_argument("--units", type=str, default="m", choices=["cm", "m"])
    p.add_argument("--radar-x-min", type=float, default=0.0)
    p.add_argument("--radar-x-max", type=float, default=150.0)
    p.add_argument("--radar-speed-min", type=float, default=None)
    p.add_argument("--radar-speed-max", type=float, default=None)
    p.add_argument("--match-shrink", type=float, default=0.02)
    p.add_argument("--match-min-points", type=int, default=1)

    p.add_argument("--mast3r-repo", type=Path, default=Path("third_party/mast3r"))
    p.add_argument(
        "--mast3r-model-name",
        type=str,
        default="naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric",
    )
    p.add_argument("--mast3r-weights", type=Path, default=None)
    p.add_argument("--mast3r-image-size", type=int, default=512)
    p.add_argument("--mast3r-device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--mast3r-min-conf", type=float, default=0.5)
    p.add_argument("--mast3r-neighbor", type=str, default="next", choices=["next", "prev", "both"])

    p.add_argument(
        "--box-inner-scale",
        type=float,
        default=0.72,
        help="Sample the central part of the 2D vehicle box to reduce background and road pixels.",
    )
    p.add_argument("--depth-trim", type=float, default=0.15, help="Trim this fraction from near/far depth tails.")
    p.add_argument("--min-image-points", type=int, default=80)
    p.add_argument("--radar-depth-tolerance-m", type=float, default=2.0)
    p.add_argument("--min-high-confidence-score", type=float, default=0.72)
    return p.parse_args()


def _inner_box(box: List[int], scale: float, image_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
    h, w = image_shape
    x1, y1, x2, y2 = [float(v) for v in box]
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    bw = max(1.0, (x2 - x1) * float(scale))
    bh = max(1.0, (y2 - y1) * float(scale))
    ix1 = int(max(0, min(w - 1, round(cx - bw * 0.5))))
    iy1 = int(max(0, min(h - 1, round(cy - bh * 0.5))))
    ix2 = int(max(0, min(w, round(cx + bw * 0.5))))
    iy2 = int(max(0, min(h, round(cy + bh * 0.5))))
    return ix1, iy1, ix2, iy2


def _scale_box(box: List[int], src_shape: Tuple[int, int], dst_shape: Tuple[int, int]) -> List[int]:
    src_h, src_w = src_shape
    dst_h, dst_w = dst_shape
    sx = float(dst_w) / max(float(src_w), 1.0)
    sy = float(dst_h) / max(float(src_h), 1.0)
    x1, y1, x2, y2 = box
    return [
        int(round(float(x1) * sx)),
        int(round(float(y1) * sy)),
        int(round(float(x2) * sx)),
        int(round(float(y2) * sy)),
    ]


def _robust_image_center_from_pointmap(
    pts3d_m: np.ndarray,
    conf: np.ndarray,
    box_xyxy: List[int],
    min_conf: float,
    inner_scale: float,
    depth_trim: float,
    min_points: int,
) -> Tuple[Optional[np.ndarray], Dict]:
    h, w = pts3d_m.shape[:2]
    x1, y1, x2, y2 = _inner_box(box_xyxy, inner_scale, (h, w))
    if x2 <= x1 or y2 <= y1:
        return None, {"status": "empty_inner_box", "inner_box_xyxy": [x1, y1, x2, y2]}

    roi_pts = pts3d_m[y1:y2, x1:x2, :].reshape(-1, 3)
    roi_conf = conf[y1:y2, x1:x2].reshape(-1)
    yy, xx = np.mgrid[y1:y2, x1:x2]
    roi_uv = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1).astype(np.float32)
    ok = np.isfinite(roi_pts).all(axis=1) & np.isfinite(roi_conf) & (roi_conf >= float(min_conf))
    pts = roi_pts[ok]
    cvals = roi_conf[ok]
    pix = roi_uv[ok]
    if len(pts) < int(min_points):
        return None, {
            "status": "too_few_image_points",
            "inner_box_xyxy": [x1, y1, x2, y2],
            "image_points": int(len(pts)),
        }

    z = pts[:, 2]
    finite_depth = np.isfinite(z)
    pts = pts[finite_depth]
    cvals = cvals[finite_depth]
    pix = pix[finite_depth]
    if len(pts) < int(min_points):
        return None, {
            "status": "too_few_finite_depth_points",
            "inner_box_xyxy": [x1, y1, x2, y2],
            "image_points": int(len(pts)),
        }

    trim = max(0.0, min(0.45, float(depth_trim)))
    if trim > 0.0 and len(pts) >= 10:
        lo, hi = np.quantile(pts[:, 2], [trim, 1.0 - trim])
        keep = (pts[:, 2] >= lo) & (pts[:, 2] <= hi)
        pts = pts[keep]
        cvals = cvals[keep]
        pix = pix[keep]

    if len(pts) < int(min_points):
        return None, {
            "status": "too_few_trimmed_points",
            "inner_box_xyxy": [x1, y1, x2, y2],
            "image_points": int(len(pts)),
        }

    center = np.median(pts, axis=0).astype(np.float32)
    q10, q50, q90 = np.quantile(pts[:, 2], [0.10, 0.50, 0.90])
    stats = {
        "status": "ok",
        "inner_box_xyxy": [x1, y1, x2, y2],
        "image_points": int(len(pts)),
        "image_conf_median": float(np.median(cvals)),
        "image_depth_median": float(q50),
        "image_depth_p10": float(q10),
        "image_depth_p90": float(q90),
        "pointmap_center_uv": [float(v) for v in np.median(pix, axis=0).tolist()],
    }
    return center, stats


def _project_radar(
    frame: RadarFrame,
    args: argparse.Namespace,
    calib: Dict[str, np.ndarray],
    axis_M: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    pts_xyz_raw = frame.points_xyz.astype(np.float32).copy()
    if len(pts_xyz_raw) == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    speed_mps = None if frame.speed_mps is None else frame.speed_mps.astype(np.float32).copy()
    unit_to_m = 1.0 if args.units == "m" else 0.01
    x_m = pts_xyz_raw[:, 0] * unit_to_m
    keep = np.isfinite(pts_xyz_raw).all(axis=1) & (x_m >= args.radar_x_min) & (x_m <= args.radar_x_max)
    if speed_mps is not None and len(speed_mps) == len(pts_xyz_raw):
        if args.radar_speed_min is not None:
            keep &= np.isfinite(speed_mps) & (speed_mps >= args.radar_speed_min)
        if args.radar_speed_max is not None:
            keep &= np.isfinite(speed_mps) & (speed_mps <= args.radar_speed_max)

    pts_xyz = pts_xyz_raw[keep]
    if len(pts_xyz) == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
    pts_xyz_cm = pts_xyz * 100.0 if args.units == "m" else pts_xyz
    uv, valid_mask = project_points(pts_xyz_cm, calib["K"], calib["R"], calib["t"], axis_M, calib.get("dist"))
    return uv, pts_xyz_cm[valid_mask]


def _radar_validation(
    pts_idx: np.ndarray,
    pts_xyz_proj_cm: np.ndarray,
    image_center_m: np.ndarray,
    tolerance_m: float,
) -> Dict:
    if len(pts_idx) == 0 or len(pts_xyz_proj_cm) == 0:
        return {
            "matched_points": int(len(pts_idx)),
            "radar_surface_center_m": None,
            "radar_depth_delta_m": None,
            "radar_depth_consistent": None,
        }

    radar_surface_m = np.median(pts_xyz_proj_cm[pts_idx], axis=0).astype(np.float32) / 100.0
    delta = float(abs(float(image_center_m[2]) - float(radar_surface_m[2])))
    return {
        "matched_points": int(len(pts_idx)),
        "radar_surface_center_m": [float(v) for v in radar_surface_m.tolist()],
        "radar_depth_delta_m": delta,
        "radar_depth_consistent": bool(delta <= float(tolerance_m)),
    }


def _confidence_score(det_conf: float, image_stats: Dict, radar_stats: Dict, min_points: int) -> Tuple[float, List[str]]:
    reasons: List[str] = []
    score = 0.0
    score += 0.30 * max(0.0, min(1.0, float(det_conf)))

    image_points = int(image_stats.get("image_points") or 0)
    point_score = max(0.0, min(1.0, image_points / max(float(min_points) * 4.0, 1.0)))
    score += 0.25 * point_score
    if point_score < 0.5:
        reasons.append("few_image_points")

    img_conf = image_stats.get("image_conf_median")
    if img_conf is not None:
        conf_score = max(0.0, min(1.0, float(img_conf)))
        score += 0.25 * conf_score
        if conf_score < 0.5:
            reasons.append("low_image_geometry_conf")
    else:
        reasons.append("missing_image_geometry_conf")

    radar_ok = radar_stats.get("radar_depth_consistent")
    if radar_ok is True:
        score += 0.15
    elif radar_ok is False:
        reasons.append("radar_depth_mismatch")
    else:
        score += 0.05
        reasons.append("no_radar_depth_check")

    spread = None
    if image_stats.get("image_depth_p90") is not None and image_stats.get("image_depth_p10") is not None:
        spread = float(image_stats["image_depth_p90"]) - float(image_stats["image_depth_p10"])
    if spread is not None and spread < 4.0:
        score += 0.05
    elif spread is not None:
        reasons.append("wide_image_depth_spread")

    return float(max(0.0, min(1.0, score))), reasons


def main() -> None:
    args = parse_args()
    if not args.images_dir.exists():
        raise FileNotFoundError(f"Images dir not found: {args.images_dir}")
    visual_only = bool(args.visual_only or args.radar_csv is None)
    if not visual_only and not args.radar_csv.exists():
        raise FileNotFoundError(f"Radar CSV not found: {args.radar_csv}")

    images = sorted(args.images_dir.glob(args.pattern))
    if args.limit > 0:
        images = images[: args.limit]
    if not images:
        raise RuntimeError("No images found")

    frames: List[RadarFrame] = []
    if not visual_only:
        frames_raw = load_radar_frames(args.radar_csv)
        frames = [RadarFrame(f.timestamp_sec, f.points_xyz, f.speed_mps) for f in frames_raw]
        if not frames:
            raise RuntimeError("No radar frames parsed")

    if args.auto_offset and frames:
        first_ts = next((parse_image_timestamp(p) for p in images if parse_image_timestamp(p) is not None), None)
        if first_ts is not None:
            args.time_offset = float(frames[0].timestamp_sec - first_ts)

    det_model = YOLO(args.model)
    axis_M = parse_axis_map(args.radar_axis)
    nearest_idx = 0
    device = pick_device(args.mast3r_device)
    predict_points3d, mast3r_source = build_mast3r_predictor(
        repo=args.mast3r_repo,
        model_name=args.mast3r_model_name,
        weights=args.mast3r_weights,
        image_size=args.mast3r_image_size,
        device=device,
    )

    out_frames: List[Dict] = []
    for idx, img_path in enumerate(images, start=1):
        ts = parse_image_timestamp(img_path)
        base_record: Dict = {
            "frame_index": idx,
            "image": str(img_path),
            "image_timestamp": None if ts is None else float(ts),
            "status": "unknown",
        }
        if ts is None and not visual_only:
            base_record["status"] = "no_image_timestamp"
            out_frames.append(base_record)
            continue

        frame = None
        ts_radar = None
        if not visual_only:
            ts_radar = float(ts + args.time_offset)
            frame, nearest_idx = find_nearest_frame(frames, ts_radar, nearest_idx)
            if frame is None or abs(frame.timestamp_sec - ts_radar) > args.max_dt:
                base_record["status"] = "no_radar_match"
                out_frames.append(base_record)
                continue

        img = cv2.imread(str(img_path))
        if img is None:
            base_record["status"] = "image_read_failed"
            out_frames.append(base_record)
            continue
        h, w = img.shape[:2]

        calib = default_calibration(w, h)
        if args.calib_json is not None:
            calib = load_calibration_override(args.calib_json, calib)
        calib["K"] = scale_intrinsics(calib["K"], w, h, args.k_scale, args.calib_width, args.calib_height)

        if frame is not None:
            uv, pts_xyz_proj = _project_radar(frame, args, calib, axis_M)
        else:
            uv = np.zeros((0, 2), dtype=np.float32)
            pts_xyz_proj = np.zeros((0, 3), dtype=np.float32)

        cur_i = idx - 1
        nbr_i = choose_neighbor_idx(cur_i, len(images), args.mast3r_neighbor)
        try:
            pts3d_map_m, conf_map = predict_points3d(images[cur_i], images[nbr_i])
        except Exception as e:
            base_record["status"] = "image_geometry_failed"
            base_record["error"] = str(e)
            if frame is not None:
                base_record["radar_timestamp"] = float(frame.timestamp_sec)
            out_frames.append(base_record)
            continue

        yolo = det_model(img, verbose=False, conf=args.conf, imgsz=args.imgsz)[0]
        boxes: List[List[int]] = []
        confs: List[float] = []
        labels: List[str] = []
        if yolo.boxes is not None:
            for b in yolo.boxes:
                cls_id = int(b.cls[0])
                if cls_id not in VEHICLE_CLASSES:
                    continue
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                boxes.append([x1, y1, x2, y2])
                confs.append(float(b.conf[0]))
                labels.append(VEHICLE_CLASSES[cls_id])

        if not boxes:
            base_record["status"] = "no_vehicle_detection"
            if frame is not None:
                base_record["radar_timestamp"] = float(frame.timestamp_sec)
            out_frames.append(base_record)
            continue

        if len(uv) > 0:
            matches = match_points_to_boxes(uv, np.asarray(boxes, dtype=np.float32), args.match_shrink)
        else:
            matches = [[] for _ in boxes]

        detections: List[Dict] = []
        for i_box, box in enumerate(boxes):
            pointmap_box = _scale_box(box, src_shape=(h, w), dst_shape=pts3d_map_m.shape[:2])
            center_m, image_stats = _robust_image_center_from_pointmap(
                pts3d_m=pts3d_map_m,
                conf=conf_map,
                box_xyxy=pointmap_box,
                min_conf=args.mast3r_min_conf,
                inner_scale=args.box_inner_scale,
                depth_trim=args.depth_trim,
                min_points=args.min_image_points,
            )
            image_stats["pointmap_box_xyxy"] = [int(v) for v in pointmap_box]
            image_stats["pointmap_shape_hw"] = [int(pts3d_map_m.shape[0]), int(pts3d_map_m.shape[1])]
            if image_stats.get("pointmap_center_uv") is not None:
                pm_u, pm_v = image_stats["pointmap_center_uv"]
                image_stats["original_center_uv"] = [
                    float(pm_u) * float(w) / max(float(pts3d_map_m.shape[1]), 1.0),
                    float(pm_v) * float(h) / max(float(pts3d_map_m.shape[0]), 1.0),
                ]
            if center_m is None:
                detections.append(
                    {
                        "status": "no_image_center",
                        "box_xyxy": [int(v) for v in box],
                        "label": labels[i_box],
                        "det_conf": float(confs[i_box]),
                        "image_geometry": image_stats,
                    }
                )
                continue

            pts_idx = np.asarray(matches[i_box], dtype=np.int32)
            if len(pts_idx) < args.match_min_points:
                pts_idx = np.zeros((0,), dtype=np.int32)
            radar_stats = _radar_validation(pts_idx, pts_xyz_proj, center_m, args.radar_depth_tolerance_m)
            score, low_conf_reasons = _confidence_score(
                det_conf=float(confs[i_box]),
                image_stats=image_stats,
                radar_stats=radar_stats,
                min_points=args.min_image_points,
            )
            center_uv = image_stats.get("original_center_uv")
            geometry_projected_uv = project_cam_point_to_uv(center_m, calib["K"])
            det = {
                "status": "ok",
                "label_source": "visual_only_pseudo_gt" if visual_only else "image_dominant_pseudo_gt",
                "mode": "mast3r_image_pointmap",
                "box_xyxy": [int(v) for v in box],
                "label": labels[i_box],
                "det_conf": float(confs[i_box]),
                "pseudo_gt_vehicle_center_m": [float(v) for v in center_m.tolist()],
                "pseudo_gt_center_uv": center_uv,
                "geometry_projected_uv": geometry_projected_uv,
                "image_geometry": image_stats,
                "radar_validation": radar_stats,
                "label_confidence": score,
                "is_high_confidence": bool(score >= float(args.min_high_confidence_score)),
                "low_confidence_reasons": low_conf_reasons,
            }
            detections.append(det)

        valid = [d for d in detections if d.get("status") == "ok"]
        if not valid:
            base_record["status"] = "no_valid_image_center"
            if frame is not None and ts_radar is not None:
                base_record["radar_timestamp"] = float(frame.timestamp_sec)
                base_record["time_delta_sec"] = float(ts_radar - frame.timestamp_sec)
            base_record["detections"] = detections
            out_frames.append(base_record)
            continue

        primary = max(valid, key=lambda d: float(d.get("label_confidence") or 0.0))
        frame_out = {
            "frame_index": idx,
            "image": str(img_path),
            "image_timestamp": None if ts is None else float(ts),
            "status": "ok",
            "selected_label_source": "visual_only_pseudo_gt" if visual_only else "image_dominant_pseudo_gt",
            "selected_box_xyxy": primary.get("box_xyxy"),
            "selected_label": primary.get("label"),
            "selected_conf": primary.get("label_confidence"),
            "pseudo_gt_vehicle_center_m": primary.get("pseudo_gt_vehicle_center_m"),
            "pseudo_gt_center_uv": primary.get("pseudo_gt_center_uv"),
            "radar_validation": primary.get("radar_validation"),
            "detections": detections,
        }
        if frame is not None and ts_radar is not None:
            frame_out["radar_timestamp"] = float(frame.timestamp_sec)
            frame_out["time_delta_sec"] = float(ts_radar - frame.timestamp_sec)
        out_frames.append(frame_out)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "backend": "image_dominant_pseudo_gt",
            "primary_geometry_backend": "mast3r",
            "label_definition": (
                "Vehicle 3D center candidate is estimated from image geometry inside the vehicle 2D box."
                if visual_only
                else "Vehicle 3D center candidate is estimated from image geometry inside the vehicle 2D box; radar is validation only."
            ),
            "mast3r_source": mast3r_source,
            "mast3r_repo": str(args.mast3r_repo),
            "mast3r_image_size": int(args.mast3r_image_size),
            "mast3r_device": device,
            "mast3r_min_conf": float(args.mast3r_min_conf),
            "mast3r_neighbor": args.mast3r_neighbor,
            "box_inner_scale": float(args.box_inner_scale),
            "depth_trim": float(args.depth_trim),
            "min_image_points": int(args.min_image_points),
            "radar_role": "disabled_visual_only" if visual_only else "weak_validation_only",
            "radar_depth_tolerance_m": float(args.radar_depth_tolerance_m),
            "images_dir": str(args.images_dir),
            "radar_csv": None if args.radar_csv is None else str(args.radar_csv),
            "time_offset": float(args.time_offset),
            "max_dt": float(args.max_dt),
            "radar_axis": args.radar_axis,
            "units": args.units,
            "detector_model": str(args.model),
            "detector_conf": float(args.conf),
            "imgsz": int(args.imgsz),
            "num_images": len(images),
            "num_frames_out": len(out_frames),
        },
        "frames": out_frames,
    }
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    ok = sum(1 for x in out_frames if str(x.get("status", "")) == "ok")
    high = sum(
        1
        for fr in out_frames
        for det in fr.get("detections", [])
        if det.get("status") == "ok" and det.get("is_high_confidence") is True
    )
    print(f"Done (image pseudo-GT). ok_frames={ok}/{len(out_frames)} high_conf_dets={high} -> {args.output_json}")


if __name__ == "__main__":
    sys.exit(main())
