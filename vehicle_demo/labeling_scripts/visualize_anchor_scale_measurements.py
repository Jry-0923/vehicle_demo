#!/usr/bin/env python3
"""Render anchor scale measurements for manual outlier review."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Draw anchor endpoint lines and scale measurements on source images.")
    p.add_argument("--measurements-csv", type=Path, default=Path("outputs/anchor_scale_review_alt_angle/anchor_scale_measurements_railing_only.csv"))
    p.add_argument("--summary-json", type=Path, default=Path("outputs/anchor_scale_review_alt_angle/anchor_scale_summary_railing_only.json"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/anchor_scale_review_alt_angle/measurement_review_railing_only"))
    p.add_argument("--only-outliers", action="store_true")
    p.add_argument("--max-image-width", type=int, default=1600)
    return p.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _resize_to_width(img, max_width: int):
    h, w = img.shape[:2]
    if max_width <= 0 or w <= max_width:
        return img.copy(), 1.0
    scale = float(max_width) / float(w)
    out = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    return out, scale


def _float(row: dict[str, str], key: str) -> float | None:
    try:
        return float(row.get(key, ""))
    except Exception:
        return None


def _int(row: dict[str, str], key: str) -> int | None:
    try:
        return int(round(float(row.get(key, ""))))
    except Exception:
        return None


def _draw_text(img, lines: list[str]) -> None:
    y = 30
    for line in lines:
        cv2.putText(img, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 0, 0), 5)
        cv2.putText(img, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2)
        y += 31


def _render(row: dict[str, str], out_path: Path, is_outlier: bool, summary: dict) -> bool:
    image_path = Path(row.get("image", ""))
    img = cv2.imread(str(image_path))
    if img is None:
        return False
    shown, scale = _resize_to_width(img, int(summary.get("max_image_width", 1600) or 1600))

    pts = []
    for u_key, v_key in (("bottom_u", "bottom_v"), ("top_u", "top_v")):
        u = _int(row, u_key)
        v = _int(row, v_key)
        if u is None or v is None:
            return False
        pts.append((int(round(u * scale)), int(round(v * scale))))

    bottom, top = pts
    color = (0, 0, 255) if is_outlier else (0, 220, 0)
    cv2.line(shown, bottom, top, (255, 255, 255), 8)
    cv2.line(shown, bottom, top, color, 3)
    cv2.circle(shown, bottom, 9, (0, 255, 255), -1)
    cv2.circle(shown, top, 9, (255, 0, 255), -1)
    cv2.putText(shown, "bottom", (bottom[0] + 10, bottom[1] + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4)
    cv2.putText(shown, "bottom", (bottom[0] + 10, bottom[1] + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    cv2.putText(shown, "top", (top[0] + 10, top[1] + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4)
    cv2.putText(shown, "top", (top[0] + 10, top[1] + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 255), 2)

    pred = _float(row, "predicted_height_m")
    scale_i = _float(row, "scale")
    true_h = _float(row, "true_height_m")
    _draw_text(
        shown,
        [
            f"{row.get('anchor_id')}  {'OUTLIER' if is_outlier else 'INLIER'}  image={image_path.name}",
            f"true_h={true_h:.3f}m  predicted_h={pred:.3f}m  scale_i={scale_i:.3f}" if pred is not None and scale_i is not None and true_h is not None else "measurement unavailable",
            f"global scale_median={summary.get('scale_median')}  inliers={summary.get('inlier_measurements')}/{summary.get('valid_measurements')}",
        ],
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return cv2.imwrite(str(out_path), shown)


def main() -> None:
    args = parse_args()
    rows = _read_csv(args.measurements_csv)
    summary = json.load(args.summary_json.open("r", encoding="utf-8")) if args.summary_json.exists() else {}
    summary["max_image_width"] = args.max_image_width
    outliers = set(summary.get("bad_anchor_outliers") or [])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for old in args.output_dir.glob("*.jpg"):
        old.unlink()

    rendered = 0
    for row in rows:
        anchor_id = row.get("anchor_id", "")
        is_outlier = anchor_id in outliers
        if args.only_outliers and not is_outlier:
            continue
        prefix = "OUT" if is_outlier else "IN"
        out_path = args.output_dir / f"{prefix}_{anchor_id}_{Path(row.get('image','')).name}"
        if _render(row, out_path, is_outlier, summary):
            rendered += 1

    print(f"Rendered {rendered} review images to {args.output_dir}")


if __name__ == "__main__":
    main()
