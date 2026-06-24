#!/usr/bin/env python3
"""Estimate MASt3R metric scale from fixed-height scene anchors."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent
for _path in (_SCRIPT_DIR, _ROOT / "ultralytics" / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from export_vehicle_centers_mast3r import build_mast3r_predictor, choose_neighbor_idx, pick_device


FIELDS = [
    "anchor_id",
    "image",
    "anchor_type",
    "bottom_u",
    "bottom_v",
    "top_u",
    "top_v",
    "true_height_m",
    "predicted_height_m",
    "height_mode",
    "delta_x_m",
    "delta_y_m",
    "delta_z_m",
    "scale",
    "point_conf_bottom",
    "point_conf_top",
    "status",
    "outlier",
    "notes",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Estimate a global MASt3R scale correction from fixed-height anchors.")
    p.add_argument("--anchors-csv", type=Path, default=Path("outputs/anchor_scale_review/anchors.csv"))
    p.add_argument("--output-json", type=Path, default=Path("outputs/anchor_scale_review/anchor_scale_summary.json"))
    p.add_argument("--output-csv", type=Path, default=Path("outputs/anchor_scale_review/anchor_scale_measurements.csv"))
    p.add_argument("--mast3r-repo", type=Path, default=Path("third_party/mast3r"))
    p.add_argument("--mast3r-model-name", type=str, default="naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric")
    p.add_argument("--mast3r-weights", type=Path, default=None)
    p.add_argument("--mast3r-image-size", type=int, default=512)
    p.add_argument("--mast3r-device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--mast3r-neighbor", type=str, default="next", choices=["next", "prev", "both"])
    p.add_argument("--sample-radius", type=int, default=2, help="Median sample radius in pointmap pixels around each anchor endpoint.")
    p.add_argument("--min-conf", type=float, default=0.5)
    p.add_argument("--outlier-rel-threshold", type=float, default=0.25)
    p.add_argument(
        "--height-mode",
        choices=["euclidean", "vertical"],
        default="euclidean",
        help="euclidean uses norm(top-bottom); vertical uses the pointmap/camera vertical component only.",
    )
    return p.parse_args()


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _float(row: dict[str, str], key: str) -> float | None:
    val = str(row.get(key, "")).strip()
    if not val:
        return None
    try:
        out = float(val)
    except Exception:
        return None
    return out if np.isfinite(out) else None


def _complete_anchor(row: dict[str, str]) -> bool:
    required = ("image", "bottom_u", "bottom_v", "top_u", "top_v", "true_height_m")
    return all(_float(row, k) is not None if k != "image" else bool(row.get(k)) for k in required)


def _image_list_for(path: Path) -> list[Path]:
    return sorted(path.parent.glob("*.jpg"))


def _sample_point(
    pts3d: np.ndarray,
    conf: np.ndarray,
    uv_orig: tuple[float, float],
    orig_wh: tuple[int, int],
    radius: int,
    min_conf: float,
) -> tuple[np.ndarray | None, float | None, str]:
    orig_w, orig_h = orig_wh
    h, w = pts3d.shape[:2]
    u = float(uv_orig[0]) * float(w) / max(float(orig_w), 1.0)
    v = float(uv_orig[1]) * float(h) / max(float(orig_h), 1.0)
    x = int(round(u))
    y = int(round(v))
    r = max(0, int(radius))
    x1, x2 = max(0, x - r), min(w, x + r + 1)
    y1, y2 = max(0, y - r), min(h, y + r + 1)
    if x1 >= x2 or y1 >= y2:
        return None, None, "outside_pointmap"
    pts = pts3d[y1:y2, x1:x2, :].reshape(-1, 3)
    vals = conf[y1:y2, x1:x2].reshape(-1)
    ok = np.isfinite(pts).all(axis=1) & np.isfinite(vals) & (vals >= float(min_conf))
    if not np.any(ok):
        return None, None, "no_confident_points"
    pts_ok = pts[ok]
    conf_ok = vals[ok]
    return np.median(pts_ok, axis=0).astype(np.float32), float(np.median(conf_ok)), "ok"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in FIELDS})


def main() -> None:
    args = parse_args()
    if not args.anchors_csv.exists():
        raise FileNotFoundError(f"Anchors CSV not found: {args.anchors_csv}")

    rows = [r for r in _read_rows(args.anchors_csv) if _complete_anchor(r)]
    if not rows:
        raise RuntimeError(f"No complete anchors found in {args.anchors_csv}. Fill bottom/top pixels first.")

    device = pick_device(args.mast3r_device)
    predict_points3d, mast3r_source = build_mast3r_predictor(
        repo=args.mast3r_repo,
        model_name=args.mast3r_model_name,
        weights=args.mast3r_weights,
        image_size=args.mast3r_image_size,
        device=device,
    )

    pointmap_cache: dict[str, tuple[np.ndarray, np.ndarray, tuple[int, int]]] = {}
    measurements: list[dict[str, Any]] = []
    for row in rows:
        image = Path(row["image"])
        if not image.exists():
            measurements.append({**row, "status": "image_missing"})
            continue
        if str(image) not in pointmap_cache:
            images = _image_list_for(image)
            try:
                cur_i = images.index(image)
            except ValueError:
                measurements.append({**row, "status": "image_not_in_parent_list"})
                continue
            nbr_i = choose_neighbor_idx(cur_i, len(images), args.mast3r_neighbor)
            pts3d, conf = predict_points3d(images[cur_i], images[nbr_i])
            img = cv2.imread(str(image))
            if img is None:
                measurements.append({**row, "status": "image_read_failed"})
                continue
            orig_h, orig_w = img.shape[:2]
            pointmap_cache[str(image)] = (pts3d, conf, (orig_w, orig_h))

        pts3d, conf, orig_wh = pointmap_cache[str(image)]
        bottom = (_float(row, "bottom_u"), _float(row, "bottom_v"))
        top = (_float(row, "top_u"), _float(row, "top_v"))
        true_height = _float(row, "true_height_m")
        if bottom[0] is None or bottom[1] is None or top[0] is None or top[1] is None or true_height is None:
            measurements.append({**row, "status": "incomplete"})
            continue

        p0, c0, s0 = _sample_point(pts3d, conf, (bottom[0], bottom[1]), orig_wh, args.sample_radius, args.min_conf)
        p1, c1, s1 = _sample_point(pts3d, conf, (top[0], top[1]), orig_wh, args.sample_radius, args.min_conf)
        out = dict(row)
        out["point_conf_bottom"] = c0
        out["point_conf_top"] = c1
        out["outlier"] = False
        if p0 is None or p1 is None:
            out["status"] = f"endpoint_failed:{s0},{s1}"
            measurements.append(out)
            continue
        delta = (p1 - p0).astype(np.float32)
        out["height_mode"] = args.height_mode
        out["delta_x_m"] = float(delta[0])
        out["delta_y_m"] = float(delta[1])
        out["delta_z_m"] = float(delta[2])
        if args.height_mode == "vertical":
            predicted = float(abs(delta[1]))
        else:
            predicted = float(np.linalg.norm(delta))
        out["predicted_height_m"] = predicted
        if predicted <= 1e-6:
            out["status"] = "zero_predicted_height"
            measurements.append(out)
            continue
        out["scale"] = float(true_height / predicted)
        out["status"] = "ok"
        measurements.append(out)

    ok = [m for m in measurements if m.get("status") == "ok" and m.get("scale") not in ("", None)]
    if not ok:
        _write_csv(args.output_csv, measurements)
        raise RuntimeError("No valid anchor measurements. Check anchor pixels and confidence threshold.")

    scales = np.asarray([float(m["scale"]) for m in ok], dtype=np.float32)
    median = float(np.median(scales))
    rel_dev = np.abs(scales - median) / max(abs(median), 1e-6)
    inlier_mask = rel_dev <= float(args.outlier_rel_threshold)
    for m, is_out in zip(ok, ~inlier_mask):
        m["outlier"] = bool(is_out)
    inlier_scales = scales[inlier_mask]

    summary = {
        "task": "anchor_scale_estimation",
        "anchors_csv": str(args.anchors_csv),
        "mast3r_source": mast3r_source,
        "device": device,
        "height_mode": args.height_mode,
        "valid_measurements": int(len(ok)),
        "inlier_measurements": int(len(inlier_scales)),
        "scale_median": float(np.median(inlier_scales)) if len(inlier_scales) else median,
        "scale_mean": float(np.mean(inlier_scales)) if len(inlier_scales) else float(np.mean(scales)),
        "scale_std": float(np.std(inlier_scales)) if len(inlier_scales) else float(np.std(scales)),
        "scale_all_median": median,
        "outlier_rel_threshold": float(args.outlier_rel_threshold),
        "bad_anchor_outliers": [m.get("anchor_id") for m in ok if m.get("outlier")],
        "status_counts": {str(k): int(v) for k, v in __import__("collections").Counter(m.get("status") for m in measurements).items()},
    }

    _write_csv(args.output_csv, measurements)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote measurements: {args.output_csv}")


if __name__ == "__main__":
    main()
