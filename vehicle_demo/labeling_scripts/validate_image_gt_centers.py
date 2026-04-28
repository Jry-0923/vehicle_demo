import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate image-dominant pseudo-GT vehicle centers.")
    p.add_argument("--centers-json", type=Path, required=True)
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--bev-png", type=Path, required=True)
    p.add_argument("--max-frame-jump-m", type=float, default=2.0)
    p.add_argument("--high-confidence-only", action="store_true")
    p.add_argument("--bev-width", type=int, default=1200)
    p.add_argument("--bev-height", type=int, default=900)
    p.add_argument("--bev-margin", type=int, default=60)
    return p.parse_args()


def _as_point_m(values) -> Optional[np.ndarray]:
    if not isinstance(values, list) or len(values) != 3:
        return None
    pt = np.asarray(values, dtype=np.float32)
    if not np.isfinite(pt).all():
        return None
    return pt


def _iter_valid_detections(obj: Dict, high_conf_only: bool) -> List[Dict]:
    rows: List[Dict] = []
    for fr in obj.get("frames", []):
        detections = fr.get("detections") or []
        for di, det in enumerate(detections):
            if det.get("status") != "ok":
                continue
            if high_conf_only and det.get("is_high_confidence") is not True:
                continue
            center = _as_point_m(det.get("pseudo_gt_vehicle_center_m"))
            if center is None:
                continue
            rows.append(
                {
                    "frame_index": int(fr.get("frame_index", len(rows) + 1)),
                    "image": fr.get("image"),
                    "det_index": int(di),
                    "label": det.get("label"),
                    "det_conf": det.get("det_conf"),
                    "label_confidence": det.get("label_confidence"),
                    "is_high_confidence": bool(det.get("is_high_confidence")),
                    "center_m": center,
                    "radar_validation": det.get("radar_validation") or {},
                    "low_confidence_reasons": det.get("low_confidence_reasons") or [],
                }
            )
    return rows


def _selected_rows(obj: Dict, high_conf_only: bool) -> List[Dict]:
    rows: List[Dict] = []
    for fr in obj.get("frames", []):
        if high_conf_only:
            candidates = []
            for det in fr.get("detections") or []:
                if det.get("status") != "ok" or det.get("is_high_confidence") is not True:
                    continue
                center = _as_point_m(det.get("pseudo_gt_vehicle_center_m"))
                if center is None:
                    continue
                candidates.append((float(det.get("label_confidence") or 0.0), det, center))
            if not candidates:
                continue
            _, det, center = max(candidates, key=lambda x: x[0])
            rows.append(
                {
                    "frame_index": int(fr.get("frame_index", len(rows) + 1)),
                    "image": fr.get("image"),
                    "center_m": center,
                    "label_confidence": det.get("label_confidence"),
                    "radar_validation": det.get("radar_validation") or {},
                }
            )
            continue

        if fr.get("status") != "ok":
            continue
        center = _as_point_m(fr.get("pseudo_gt_vehicle_center_m"))
        if center is None:
            continue
        rows.append(
            {
                "frame_index": int(fr.get("frame_index", len(rows) + 1)),
                "image": fr.get("image"),
                "center_m": center,
                "label_confidence": fr.get("selected_conf"),
                "radar_validation": fr.get("radar_validation") or {},
            }
        )
    return rows


def _jump_stats(rows: List[Dict], max_jump_m: float) -> Dict:
    jumps: List[Dict] = []
    for a, b in zip(rows, rows[1:]):
        dist = float(np.linalg.norm(b["center_m"] - a["center_m"]))
        jumps.append(
            {
                "from_frame": int(a["frame_index"]),
                "to_frame": int(b["frame_index"]),
                "jump_m": dist,
                "is_large_jump": bool(dist > float(max_jump_m)),
            }
        )
    vals = [x["jump_m"] for x in jumps]
    return {
        "max_frame_jump_m": float(max(vals)) if vals else None,
        "median_frame_jump_m": float(np.median(vals)) if vals else None,
        "large_jump_count": int(sum(1 for x in jumps if x["is_large_jump"])),
        "large_jumps": [x for x in jumps if x["is_large_jump"]],
    }


def _draw_bev(rows: List[Dict], out_png: Path, width: int, height: int, margin: int) -> None:
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    if not rows:
        cv2.putText(img, "No valid centers", (margin, margin), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_png), img)
        return

    pts = np.asarray([r["center_m"] for r in rows], dtype=np.float32)
    xs = pts[:, 0]
    zs = pts[:, 2]
    x_min, x_max = float(np.min(xs)), float(np.max(xs))
    z_min, z_max = float(np.min(zs)), float(np.max(zs))
    if abs(x_max - x_min) < 1e-3:
        x_min -= 1.0
        x_max += 1.0
    if abs(z_max - z_min) < 1e-3:
        z_min -= 1.0
        z_max += 1.0

    def to_px(x: float, z: float) -> Tuple[int, int]:
        px = margin + (x - x_min) / (x_max - x_min) * max(width - 2 * margin, 1)
        py = height - margin - (z - z_min) / (z_max - z_min) * max(height - 2 * margin, 1)
        return int(round(px)), int(round(py))

    for gx in np.linspace(x_min, x_max, 7):
        x0, _ = to_px(float(gx), z_min)
        cv2.line(img, (x0, margin), (x0, height - margin), (230, 230, 230), 1)
        cv2.putText(img, f"x={gx:.1f}", (x0 + 4, height - margin + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)
    for gz in np.linspace(z_min, z_max, 7):
        _, y0 = to_px(x_min, float(gz))
        cv2.line(img, (margin, y0), (width - margin, y0), (230, 230, 230), 1)
        cv2.putText(img, f"z={gz:.1f}", (8, y0 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)

    prev = None
    for r in rows:
        x, _, z = [float(v) for v in r["center_m"]]
        p = to_px(x, z)
        if prev is not None:
            cv2.line(img, prev, p, (180, 180, 180), 1)
        color = (0, 150, 0) if r.get("is_high_confidence") else (0, 140, 255)
        cv2.circle(img, p, 5, color, -1)
        cv2.putText(img, str(r["frame_index"]), (p[0] + 7, p[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        prev = p

    cv2.putText(
        img,
        "BEV validation: x lateral, z depth, green=high confidence",
        (margin, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (0, 0, 0),
        2,
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), img)


def main() -> None:
    args = parse_args()
    if not args.centers_json.exists():
        raise FileNotFoundError(f"JSON not found: {args.centers_json}")
    with args.centers_json.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    rows = _iter_valid_detections(obj, args.high_confidence_only)
    selected = _selected_rows(obj, args.high_confidence_only)
    radar_stats = [r["radar_validation"] for r in rows]
    radar_checked = [x for x in radar_stats if x.get("radar_depth_consistent") is not None]
    radar_consistent = [x for x in radar_checked if x.get("radar_depth_consistent") is True]
    high = [r for r in rows if r.get("is_high_confidence")]
    conf_vals = [float(r["label_confidence"]) for r in rows if r.get("label_confidence") is not None]

    summary = {
        "input_json": str(args.centers_json),
        "valid_detections": int(len(rows)),
        "selected_frames": int(len(selected)),
        "high_confidence_detections": int(len(high)),
        "high_confidence_ratio": float(len(high) / len(rows)) if rows else None,
        "mean_label_confidence": float(np.mean(conf_vals)) if conf_vals else None,
        "median_label_confidence": float(np.median(conf_vals)) if conf_vals else None,
        "radar_checked_detections": int(len(radar_checked)),
        "radar_consistent_detections": int(len(radar_consistent)),
        "radar_consistency_ratio": float(len(radar_consistent) / len(radar_checked)) if radar_checked else None,
        "selected_center_temporal": _jump_stats(selected, args.max_frame_jump_m),
        "bev_png": str(args.bev_png),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    _draw_bev(rows, args.bev_png, args.bev_width, args.bev_height, args.bev_margin)
    print(
        f"Validated {len(rows)} detections; high_conf={len(high)}; "
        f"radar_consistent={len(radar_consistent)}/{len(radar_checked)} -> {args.output_json}, {args.bev_png}"
    )


if __name__ == "__main__":
    main()
