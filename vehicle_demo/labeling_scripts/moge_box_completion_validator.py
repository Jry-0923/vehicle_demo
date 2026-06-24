import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


DEFAULT_PRIORS = {
    "car": {"length_m": 4.6, "width_m": 1.85, "height_m": 1.55},
    "truck": {"length_m": 8.0, "width_m": 2.5, "height_m": 3.0},
    "bus": {"length_m": 11.0, "width_m": 2.55, "height_m": 3.2},
    "van": {"length_m": 5.2, "width_m": 2.0, "height_m": 2.1},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Complete vehicle 3D box centers from MoGe-style visible point maps. "
            "MoGe points are treated as visible surface geometry, not direct center labels."
        )
    )
    p.add_argument("--centers-json", type=Path, required=True, help="Image pseudo-GT JSON with frames/detections.")
    p.add_argument("--moge-dir", type=Path, required=True, help="Directory containing per-image MoGe .npz outputs.")
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--bev-png", type=Path, required=True)
    p.add_argument("--point-key", type=str, default="auto", help="NPZ key for metric point map, or auto.")
    p.add_argument("--depth-key", type=str, default="auto", help="NPZ key for metric depth map, or auto.")
    p.add_argument("--intrinsics-key", type=str, default="auto", help="NPZ key for camera intrinsics, or auto.")
    p.add_argument("--mask-key", type=str, default="auto", help="NPZ key for valid mask/confidence, or auto.")
    p.add_argument("--pattern", type=str, default="{stem}.npz", help="MoGe file pattern using {stem} and {name}.")
    p.add_argument("--box-scale", type=float, default=0.92, help="Use a slightly shrunken 2D box to reduce background.")
    p.add_argument("--min-points", type=int, default=120)
    p.add_argument("--depth-trim", type=float, default=0.08)
    p.add_argument("--bottom-crop", type=float, default=0.04, help="Drop this lower fraction of the 2D box to reduce road pixels.")
    p.add_argument("--min-valid-ratio", type=float, default=0.03)
    p.add_argument("--max-output-detections", type=int, default=0, help="0 means all detections.")
    p.add_argument("--bev-width", type=int, default=1300)
    p.add_argument("--bev-height", type=int, default=900)
    p.add_argument("--bev-margin", type=int, default=70)
    return p.parse_args()


def _pick_key(keys, requested: str, candidates: List[str]) -> Optional[str]:
    if requested != "auto":
        return requested if requested in keys else None
    lower = {k.lower(): k for k in keys}
    for name in candidates:
        if name.lower() in lower:
            return lower[name.lower()]
    for k in keys:
        lk = k.lower()
        if any(c.lower() in lk for c in candidates):
            return k
    return None


def _load_moge_npz(path: Path, args: argparse.Namespace) -> Tuple[np.ndarray, Optional[np.ndarray], Dict]:
    data = np.load(path)
    keys = list(data.keys())
    point_key = _pick_key(keys, args.point_key, ["points", "pointmap", "pts3d", "points3d", "xyz"])
    depth_key = _pick_key(keys, args.depth_key, ["depth", "metric_depth", "depth_m"])
    intr_key = _pick_key(keys, args.intrinsics_key, ["intrinsics", "K", "cam2img", "camera_matrix"])
    mask_key = _pick_key(keys, args.mask_key, ["mask", "valid_mask", "confidence", "conf"])

    meta = {"file": str(path), "keys": keys, "point_key": point_key, "depth_key": depth_key, "intrinsics_key": intr_key, "mask_key": mask_key}
    mask = None
    if mask_key is not None:
        mask_arr = np.asarray(data[mask_key])
        if mask_arr.ndim >= 2:
            mask = mask_arr.squeeze()
            if mask.dtype != np.bool_:
                finite = np.isfinite(mask)
                threshold = 0.0 if np.nanmax(mask[finite]) <= 1.0 else float(np.nanmedian(mask[finite]))
                mask = finite & (mask > threshold)

    if point_key is not None:
        pts = np.asarray(data[point_key], dtype=np.float32)
        if pts.ndim == 3 and pts.shape[-1] == 3:
            return pts, mask, meta
        if pts.ndim == 3 and pts.shape[0] == 3:
            return np.moveaxis(pts, 0, -1).astype(np.float32), mask, meta
        raise ValueError(f"Unsupported point map shape in {path}: {pts.shape}")

    if depth_key is None or intr_key is None:
        raise ValueError(f"{path} needs either a point map key or depth+intrinsics keys; keys={keys}")

    depth = np.asarray(data[depth_key], dtype=np.float32).squeeze()
    K = np.asarray(data[intr_key], dtype=np.float32)
    if K.shape != (3, 3):
        raise ValueError(f"Unsupported intrinsics shape in {path}: {K.shape}")
    h, w = depth.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    z = depth
    x = (xx - K[0, 2]) / max(float(K[0, 0]), 1e-6) * z
    y = (yy - K[1, 2]) / max(float(K[1, 1]), 1e-6) * z
    pts = np.stack([x, y, z], axis=-1).astype(np.float32)
    return pts, mask, meta


def _moge_path_for_image(moge_dir: Path, image_path: str, pattern: str) -> Path:
    p = Path(image_path)
    return moge_dir / pattern.format(stem=p.stem, name=p.name)


def _rescale_box_to_pointmap(box: List[float], image_shape_hw: Optional[Tuple[int, int]], pointmap_shape_hw: Tuple[int, int]) -> List[float]:
    if image_shape_hw is None:
        return [float(v) for v in box]
    ih, iw = image_shape_hw
    ph, pw = pointmap_shape_hw
    if ih <= 0 or iw <= 0 or (ih == ph and iw == pw):
        return [float(v) for v in box]
    sx = float(pw) / float(iw)
    sy = float(ph) / float(ih)
    x1, y1, x2, y2 = [float(v) for v in box]
    return [x1 * sx, y1 * sy, x2 * sx, y2 * sy]


def _scaled_box(box: List[float], scale: float, shape_hw: Tuple[int, int], bottom_crop: float) -> Tuple[int, int, int, int]:
    h, w = shape_hw
    x1, y1, x2, y2 = [float(v) for v in box]
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    bw = max(1.0, (x2 - x1) * max(0.1, min(1.0, scale)))
    bh = max(1.0, (y2 - y1) * max(0.1, min(1.0, scale)))
    sx1 = int(round(max(0, min(w - 1, cx - bw * 0.5))))
    sx2 = int(round(max(0, min(w, cx + bw * 0.5))))
    sy1 = int(round(max(0, min(h - 1, cy - bh * 0.5))))
    sy2 = int(round(max(0, min(h, cy + bh * 0.5 - bh * max(0.0, min(0.35, bottom_crop))))))
    return sx1, sy1, sx2, sy2


def _vehicle_prior(label: str) -> Dict[str, float]:
    key = (label or "car").lower()
    if key in DEFAULT_PRIORS:
        return DEFAULT_PRIORS[key]
    return DEFAULT_PRIORS["car"]


def _robust_roi_points(
    pts3d: np.ndarray,
    mask: Optional[np.ndarray],
    box_xyxy: List[float],
    args: argparse.Namespace,
) -> Tuple[Optional[np.ndarray], Dict]:
    h, w = pts3d.shape[:2]
    x1, y1, x2, y2 = _scaled_box(box_xyxy, args.box_scale, (h, w), args.bottom_crop)
    if x2 <= x1 or y2 <= y1:
        return None, {"status": "empty_box", "roi_box_xyxy": [x1, y1, x2, y2]}

    roi = pts3d[y1:y2, x1:x2].reshape(-1, 3)
    ok = np.isfinite(roi).all(axis=1) & (roi[:, 2] > 0.1)
    if mask is not None and mask.shape[:2] == pts3d.shape[:2]:
        ok &= mask[y1:y2, x1:x2].reshape(-1).astype(bool)
    pts = roi[ok]
    valid_ratio = float(len(pts) / max((x2 - x1) * (y2 - y1), 1))
    if len(pts) < args.min_points or valid_ratio < args.min_valid_ratio:
        return None, {
            "status": "too_few_visible_points",
            "roi_box_xyxy": [x1, y1, x2, y2],
            "visible_points": int(len(pts)),
            "valid_ratio": valid_ratio,
        }

    trim = max(0.0, min(0.35, float(args.depth_trim)))
    if trim > 0.0 and len(pts) >= 20:
        lo, hi = np.quantile(pts[:, 2], [trim, 1.0 - trim])
        pts = pts[(pts[:, 2] >= lo) & (pts[:, 2] <= hi)]
    if len(pts) < args.min_points:
        return None, {
            "status": "too_few_trimmed_visible_points",
            "roi_box_xyxy": [x1, y1, x2, y2],
            "visible_points": int(len(pts)),
            "valid_ratio": valid_ratio,
        }

    return pts.astype(np.float32), {
        "status": "ok",
        "roi_box_xyxy": [x1, y1, x2, y2],
        "visible_points": int(len(pts)),
        "valid_ratio": valid_ratio,
        "visible_depth_median": float(np.median(pts[:, 2])),
        "visible_depth_p10": float(np.quantile(pts[:, 2], 0.10)),
        "visible_depth_p90": float(np.quantile(pts[:, 2], 0.90)),
    }


def _pca_axes_xz(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    bev = points[:, [0, 2]].astype(np.float32)
    center = np.median(bev, axis=0)
    centered = bev - center
    cov = np.cov(centered.T)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    axes = vecs[:, order].T.astype(np.float32)
    if axes[0, 1] < 0:
        axes[0] *= -1.0
    if np.linalg.det(axes) < 0:
        axes[1] *= -1.0
    return center.astype(np.float32), axes[0], axes[1]


def _complete_axis_center(
    coords: np.ndarray,
    axis: np.ndarray,
    other_axis: np.ndarray,
    origin_xz: np.ndarray,
    dim: float,
) -> Tuple[float, Dict]:
    obs_min = float(np.quantile(coords, 0.03))
    obs_max = float(np.quantile(coords, 0.97))
    span = max(0.0, obs_max - obs_min)
    midpoint = 0.5 * (obs_min + obs_max)
    coverage = min(1.0, span / max(dim, 1e-6))
    if coverage >= 0.62:
        return midpoint, {"mode": "observed_midpoint", "observed_span_m": span, "coverage": coverage}

    other_mid = 0.0
    side_min = origin_xz + axis * obs_min + other_axis * other_mid
    side_max = origin_xz + axis * obs_max + other_axis * other_mid
    min_near = float(np.linalg.norm(side_min))
    max_near = float(np.linalg.norm(side_max))
    if min_near <= max_near:
        center_coord = obs_min + dim * 0.5
        near_side = "min"
    else:
        center_coord = obs_max - dim * 0.5
        near_side = "max"
    return float(center_coord), {
        "mode": "prior_completion_from_near_side",
        "observed_span_m": span,
        "coverage": coverage,
        "near_side": near_side,
    }


def _complete_box(points: np.ndarray, label: str) -> Dict:
    prior = _vehicle_prior(label)
    length = float(prior["length_m"])
    width = float(prior["width_m"])
    visible_center_xz, axis0, axis1 = _pca_axes_xz(points)
    bev = points[:, [0, 2]].astype(np.float32)
    rel = bev - visible_center_xz
    c0 = rel @ axis0
    c1 = rel @ axis1
    span0 = float(np.quantile(c0, 0.97) - np.quantile(c0, 0.03))
    span1 = float(np.quantile(c1, 0.97) - np.quantile(c1, 0.03))

    if span0 >= width * 1.25 or span0 >= span1:
        length_axis, width_axis = axis0, axis1
        length_coords, width_coords = c0, c1
        axis_assignment = "pc0_length"
    else:
        length_axis, width_axis = axis1, axis0
        length_coords, width_coords = c1, c0
        axis_assignment = "pc1_length"

    length_center, length_meta = _complete_axis_center(length_coords, length_axis, width_axis, visible_center_xz, length)
    width_center, width_meta = _complete_axis_center(width_coords, width_axis, length_axis, visible_center_xz, width)
    center_xz = visible_center_xz + length_axis * length_center + width_axis * width_center

    corners = []
    for sl, sw in [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)]:
        p = center_xz + length_axis * (sl * length) + width_axis * (sw * width)
        corners.append([float(p[0]), float(p[1])])

    visible_center = np.median(points, axis=0).astype(np.float32)
    completed_center = np.array([center_xz[0], visible_center[1], center_xz[1]], dtype=np.float32)
    fill_penalty = (1.0 - length_meta["coverage"]) * 0.45 + (1.0 - width_meta["coverage"]) * 0.35
    points_score = min(1.0, math.log10(max(len(points), 1)) / 3.2)
    completion_conf = max(0.05, min(1.0, points_score - fill_penalty))
    return {
        "status": "ok",
        "class_size_prior_m": prior,
        "visible_surface_center_m": [float(v) for v in visible_center.tolist()],
        "completed_box_center_m": [float(v) for v in completed_center.tolist()],
        "completed_box_bev_corners_xz": corners,
        "yaw_rad_bev": float(math.atan2(float(length_axis[0]), float(length_axis[1]))),
        "axis_assignment": axis_assignment,
        "length_axis_xz": [float(v) for v in length_axis.tolist()],
        "width_axis_xz": [float(v) for v in width_axis.tolist()],
        "length_completion": length_meta,
        "width_completion": width_meta,
        "completion_confidence": float(completion_conf),
        "center_shift_from_visible_m": float(np.linalg.norm(completed_center[[0, 2]] - visible_center[[0, 2]])),
    }


def _draw_bev(rows: List[Dict], out_png: Path, width: int, height: int, margin: int) -> None:
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    ok = [r for r in rows if r.get("completion", {}).get("status") == "ok"]
    if not ok:
        cv2.putText(img, "No valid MoGe completions", (margin, margin), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_png), img)
        return

    pts = []
    for r in ok:
        comp = r["completion"]
        pts.append([comp["completed_box_center_m"][0], comp["completed_box_center_m"][2]])
        pts.extend(comp["completed_box_bev_corners_xz"])
        pts.append([comp["visible_surface_center_m"][0], comp["visible_surface_center_m"][2]])
    arr = np.asarray(pts, dtype=np.float32)
    x_min, z_min = np.min(arr, axis=0)
    x_max, z_max = np.max(arr, axis=0)
    pad_x = max(1.0, (x_max - x_min) * 0.1)
    pad_z = max(1.0, (z_max - z_min) * 0.1)
    x_min -= pad_x
    x_max += pad_x
    z_min -= pad_z
    z_max += pad_z

    def to_px(x: float, z: float) -> Tuple[int, int]:
        px = margin + (x - x_min) / max(x_max - x_min, 1e-6) * max(width - 2 * margin, 1)
        py = height - margin - (z - z_min) / max(z_max - z_min, 1e-6) * max(height - 2 * margin, 1)
        return int(round(px)), int(round(py))

    for gx in np.linspace(x_min, x_max, 7):
        p0 = to_px(float(gx), z_min)
        p1 = to_px(float(gx), z_max)
        cv2.line(img, (p0[0], margin), (p1[0], height - margin), (235, 235, 235), 1)
    for gz in np.linspace(z_min, z_max, 7):
        p0 = to_px(x_min, float(gz))
        p1 = to_px(x_max, float(gz))
        cv2.line(img, (margin, p0[1]), (width - margin, p1[1]), (235, 235, 235), 1)

    for r in ok:
        comp = r["completion"]
        corners = [to_px(x, z) for x, z in comp["completed_box_bev_corners_xz"]]
        conf = float(comp.get("completion_confidence") or 0.0)
        color = (0, 150, 0) if conf >= 0.65 else (0, 150, 255) if conf >= 0.4 else (0, 0, 220)
        for a, b in zip(corners, corners[1:] + corners[:1]):
            cv2.line(img, a, b, color, 2)
        vc = comp["visible_surface_center_m"]
        cc = comp["completed_box_center_m"]
        p_vis = to_px(float(vc[0]), float(vc[2]))
        p_ctr = to_px(float(cc[0]), float(cc[2]))
        cv2.circle(img, p_vis, 4, (180, 80, 0), -1)
        cv2.circle(img, p_ctr, 5, color, -1)
        cv2.line(img, p_vis, p_ctr, (120, 120, 120), 1)
        cv2.putText(img, f"{r['frame_index']}:{r['det_index']}", (p_ctr[0] + 7, p_ctr[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

    cv2.putText(
        img,
        "MoGe-lite BEV: blue dot=visible surface, rectangle/center=prior-completed vehicle box",
        (margin, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.66,
        (0, 0, 0),
        2,
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), img)


def _image_shape(path: str, cache: Dict[str, Optional[Tuple[int, int]]]) -> Optional[Tuple[int, int]]:
    if path in cache:
        return cache[path]
    img_path = Path(path)
    if not img_path.exists():
        cache[path] = None
        return None
    img = cv2.imread(str(img_path))
    if img is None:
        cache[path] = None
        return None
    cache[path] = (int(img.shape[0]), int(img.shape[1]))
    return cache[path]


def _summarize(rows: List[Dict]) -> Dict:
    ok = [r for r in rows if r.get("completion", {}).get("status") == "ok"]
    conf = [float(r["completion"]["completion_confidence"]) for r in ok]
    shifts = [float(r["completion"]["center_shift_from_visible_m"]) for r in ok]
    return {
        "detections_total": int(len(rows)),
        "detections_completed": int(len(ok)),
        "completion_ratio": float(len(ok) / len(rows)) if rows else None,
        "mean_completion_confidence": float(np.mean(conf)) if conf else None,
        "median_completion_confidence": float(np.median(conf)) if conf else None,
        "median_center_shift_from_visible_m": float(np.median(shifts)) if shifts else None,
        "max_center_shift_from_visible_m": float(np.max(shifts)) if shifts else None,
    }


def main() -> None:
    args = parse_args()
    with args.centers_json.open("r", encoding="utf-8") as f:
        source = json.load(f)

    moge_cache: Dict[Path, Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict, Optional[str]]] = {}
    image_shape_cache: Dict[str, Optional[Tuple[int, int]]] = {}
    rows = []
    processed = 0
    for fi, fr in enumerate(source.get("frames", [])):
        image = fr.get("image") or ""
        moge_path = _moge_path_for_image(args.moge_dir, image, args.pattern)
        if moge_path not in moge_cache:
            if not moge_path.exists():
                moge_cache[moge_path] = (None, None, {"file": str(moge_path)}, "moge_file_missing")
            else:
                try:
                    pts3d, mask, meta = _load_moge_npz(moge_path, args)
                    moge_cache[moge_path] = (pts3d, mask, meta, None)
                except Exception as exc:
                    moge_cache[moge_path] = (None, None, {"file": str(moge_path)}, f"moge_load_error: {exc}")
        pts3d, mask, moge_meta, load_error = moge_cache[moge_path]
        for di, det in enumerate(fr.get("filtered_detections") or fr.get("detections") or []):
            if args.max_output_detections and processed >= args.max_output_detections:
                break
            if det.get("status") not in (None, "ok"):
                continue
            row = {
                "frame_index": int(fr.get("frame_index", fi + 1)),
                "image": image,
                "det_index": int(di),
                "label": det.get("label"),
                "box_xyxy": det.get("box_xyxy"),
                "source_center_m": det.get("smoothed_pseudo_gt_vehicle_center_m")
                or det.get("pseudo_gt_vehicle_center_m")
                or fr.get("temporal_pseudo_gt_vehicle_center_m"),
                "moge": moge_meta,
            }
            processed += 1
            if load_error is not None:
                row["completion"] = {"status": load_error}
                rows.append(row)
                continue
            if not isinstance(det.get("box_xyxy"), list):
                row["completion"] = {"status": "missing_box_xyxy"}
                rows.append(row)
                continue
            image_shape_hw = _image_shape(image, image_shape_cache)
            pointmap_box = _rescale_box_to_pointmap(det["box_xyxy"], image_shape_hw, pts3d.shape[:2])
            row["pointmap_box_xyxy"] = [float(v) for v in pointmap_box]
            row["image_shape_hw"] = None if image_shape_hw is None else [int(image_shape_hw[0]), int(image_shape_hw[1])]
            row["pointmap_shape_hw"] = [int(pts3d.shape[0]), int(pts3d.shape[1])]
            roi_pts, roi_meta = _robust_roi_points(pts3d, mask, pointmap_box, args)
            row["visible_geometry"] = roi_meta
            if roi_pts is None:
                row["completion"] = {"status": roi_meta["status"]}
                rows.append(row)
                continue
            row["completion"] = _complete_box(roi_pts, str(det.get("label") or "car"))
            rows.append(row)
        if args.max_output_detections and processed >= args.max_output_detections:
            break

    payload = {
        "meta": {
            "task": "moge_lite_vehicle_box_completion",
            "source_centers_json": str(args.centers_json),
            "moge_dir": str(args.moge_dir),
            "assumption": "MoGe visible point maps are not treated as vehicle centers; unseen vehicle volume is completed with class size priors and BEV symmetry.",
            "class_size_priors_m": DEFAULT_PRIORS,
            "summary": _summarize(rows),
        },
        "detections": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    _draw_bev(rows, args.bev_png, args.bev_width, args.bev_height, args.bev_margin)
    s = payload["meta"]["summary"]
    print(
        f"MoGe-lite completed {s['detections_completed']}/{s['detections_total']} detections -> "
        f"{args.output_json}, {args.bev_png}"
    )


if __name__ == "__main__":
    main()
