#!/usr/bin/env python3
"""Prepare fixed-height anchor review images from all 4.15 frame folders."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


FIELDS = [
    "anchor_id",
    "source_sample_id",
    "target",
    "image",
    "image_name",
    "image_width",
    "image_height",
    "anchor_type",
    "bottom_u",
    "bottom_v",
    "top_u",
    "top_v",
    "true_height_m",
    "notes",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare alternate anchor-height review images from full frame folders.")
    p.add_argument("--frames-root", type=Path, default=Path("4.15Test/4_15_Frames"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/anchor_scale_review_alt_angle"))
    p.add_argument("--exclude-anchors-csv", type=Path, default=Path("outputs/anchor_scale_review/anchors.csv"))
    p.add_argument("--count", type=int, default=20)
    p.add_argument("--per-target", type=int, default=4)
    p.add_argument("--min-frame-gap", type=int, default=6)
    p.add_argument("--max-image-width", type=int, default=1600)
    return p.parse_args()


def _read_excluded_images(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8", newline="") as f:
        return {row.get("image", "") for row in csv.DictReader(f) if row.get("image")}


def _frame_index(path: Path) -> int:
    stem = path.stem
    # frame_<sec>_<micro>.jpg is already time-sortable, but a compact integer
    # makes spacing checks independent from file name separators.
    parts = stem.split("_")
    if len(parts) >= 3 and parts[-2].isdigit() and parts[-1].isdigit():
        return int(parts[-2]) * 1_000_000 + int(parts[-1])
    return 0


def _resize_to_width(img, max_width: int):
    h, w = img.shape[:2]
    if max_width <= 0 or w <= max_width:
        return img.copy(), 1.0
    scale = float(max_width) / float(w)
    out = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    return out, scale


def _sharpness_score(img) -> float:
    h, w = img.shape[:2]
    # Left-side railing band in this dataset. This is only used for
    # sampling candidates; annotation still uses human-clicked endpoints.
    y1, y2 = int(0.34 * h), int(0.70 * h)
    x1, x2 = int(0.02 * w), int(0.58 * w)
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    edges = cv2.Canny(gray, 60, 160)
    edge_density = float(np.count_nonzero(edges)) / float(edges.size)
    return lap_var * (0.5 + edge_density)


def _draw_text(img, lines: list[str]) -> None:
    y = 28
    for line in lines:
        cv2.putText(img, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4)
        cv2.putText(img, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
        y += 28


def _draw_roi_guides(img) -> None:
    h, w = img.shape[:2]
    left_band = (int(0.02 * w), int(0.34 * h), int(0.58 * w), int(0.70 * h))
    x1, y1, x2, y2 = left_band
    color = (0, 255, 255)
    label = "railing candidate area"
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, label, (x1 + 6, y1 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4)
    cv2.putText(img, label, (x1 + 6, y1 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2)


def _write_zoom_sheet(src, out_path: Path, max_width: int) -> None:
    shown, _ = _resize_to_width(src, max_width)
    _draw_roi_guides(shown)
    h, w = shown.shape[:2]
    crop = shown[int(0.30 * h) : int(0.72 * h), int(0.0 * w) : int(0.65 * w)]
    if crop.size:
        zoom = cv2.resize(crop, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        zh, zw = zoom.shape[:2]
        canvas = np.full((max(h, zh), w + zw, 3), 245, dtype=np.uint8)
        canvas[:h, :w] = shown
        canvas[:zh, w : w + zw] = zoom
        cv2.putText(canvas, "zoomed anchor area", (w + 16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
        cv2.imwrite(str(out_path), canvas)
    else:
        cv2.imwrite(str(out_path), shown)


def _candidate_images(frames_root: Path, excluded: set[str]) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for target_dir in sorted(frames_root.glob("Target*")):
        if not target_dir.is_dir():
            continue
        target = target_dir.name.replace("Target", "")
        for image_path in sorted(target_dir.glob("*.jpg")):
            if str(image_path.resolve()) in excluded or str(image_path) in excluded:
                continue
            img = cv2.imread(str(image_path))
            if img is None:
                continue
            candidates.append(
                {
                    "target": target,
                    "image": image_path.resolve(),
                    "image_name": image_path.name,
                    "frame_index": _frame_index(image_path),
                    "score": _sharpness_score(img),
                    "shape": img.shape[:2],
                }
            )
    return candidates


def _select(candidates: list[dict[str, object]], count: int, per_target: int, min_frame_gap: int) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    by_target: dict[str, list[dict[str, object]]] = {}
    for c in candidates:
        by_target.setdefault(str(c["target"]), []).append(c)
    for target_rows in by_target.values():
        target_rows.sort(key=lambda r: float(r["score"]), reverse=True)

    def spaced_ok(row: dict[str, object], existing: list[dict[str, object]]) -> bool:
        same_target = [r for r in existing if r["target"] == row["target"]]
        if not same_target:
            return True
        # The timestamp is in microseconds; use a broad gap to avoid near-duplicate frames.
        return all(abs(int(row["frame_index"]) - int(r["frame_index"])) >= min_frame_gap * 1_000_000 for r in same_target)

    for target in sorted(by_target):
        picked_for_target = 0
        for row in by_target[target]:
            if picked_for_target >= per_target:
                break
            if not spaced_ok(row, selected):
                continue
            selected.append(row)
            picked_for_target += 1
            if len(selected) >= count:
                return selected

    for row in sorted(candidates, key=lambda r: float(r["score"]), reverse=True):
        if row in selected or not spaced_ok(row, selected):
            continue
        selected.append(row)
        if len(selected) >= count:
            break
    return selected


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if not args.frames_root.exists():
        raise FileNotFoundError(f"Frames root not found: {args.frames_root}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = args.output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    for old in images_dir.glob("ANCH*.jpg"):
        old.unlink()

    excluded = _read_excluded_images(args.exclude_anchors_csv)
    candidates = _candidate_images(args.frames_root, excluded)
    selected = _select(candidates, args.count, args.per_target, args.min_frame_gap)
    if not selected:
        raise RuntimeError("No alternate anchor candidates selected.")

    rows: list[dict[str, object]] = []
    rendered = 0
    for i, row in enumerate(selected, start=1):
        image_path = Path(row["image"])
        img = cv2.imread(str(image_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        shown, _ = _resize_to_width(img, args.max_image_width)
        _draw_roi_guides(shown)
        _draw_text(
            shown,
            [
                f"ANCH{i:03d} Target{row['target']} score={float(row['score']):.1f} image={image_path.name}",
                "Alternate full-frame sample; yellow box highlights likely railing area.",
                "Use click labeler on the original image. Railing height is 0.92m; leave unclear rows blank.",
            ],
        )
        out_name = f"ANCH{i:03d}_T{row['target']}_{image_path.name}"
        out_path = images_dir / out_name
        cv2.imwrite(str(out_path), shown)
        _write_zoom_sheet(img, images_dir / f"ANCH{i:03d}_T{row['target']}_{image_path.stem}_zoom.jpg", args.max_image_width)
        rendered += 1

        source_sample_id = f"ALT{i:03d}"
        rows.append(
            {
                "anchor_id": f"ANCH{i:03d}_railing",
                "source_sample_id": source_sample_id,
                "target": row["target"],
                "image": str(image_path),
                "image_name": image_path.name,
                "image_width": w,
                "image_height": h,
                "anchor_type": "railing",
                "bottom_u": "",
                "bottom_v": "",
                "top_u": "",
                "top_v": "",
                "true_height_m": 0.92,
                "notes": f"railing_roi_score={float(row['score']):.3f}",
            }
        )

    _write_csv(args.output_dir / "anchors_template.csv", rows)
    _write_csv(args.output_dir / "anchors.csv", rows)
    print("Alternate anchor review package prepared.")
    print(f"Candidates scanned: {len(candidates)}")
    print(f"Review images: {images_dir} ({rendered}/{len(selected)} rendered, includes zoom copies)")
    print(f"Fill this file: {args.output_dir / 'anchors.csv'}")


if __name__ == "__main__":
    main()
