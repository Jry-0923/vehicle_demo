#!/usr/bin/env python3
"""Prepare images and a CSV template for fixed-height scene anchors."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import cv2


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
    p = argparse.ArgumentParser(description="Prepare anchor-height review images and CSV template.")
    p.add_argument("--gold-manifest", type=Path, default=Path("outputs/gold_set_review/manifest.csv"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/anchor_scale_review"))
    p.add_argument("--count", type=int, default=20)
    p.add_argument("--max-image-width", type=int, default=1600)
    return p.parse_args()


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _resize_to_width(img, max_width: int):
    h, w = img.shape[:2]
    if max_width <= 0 or w <= max_width:
        return img, 1.0
    scale = float(max_width) / float(w)
    out = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    return out, scale


def _draw_text(img, lines: list[str]) -> None:
    y = 28
    for line in lines:
        cv2.putText(img, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4)
        cv2.putText(img, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
        y += 28


def _select_rows(rows: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    used_images: set[str] = set()
    target_counts: Counter[str] = Counter()

    # First pass: keep target coverage.
    for row in rows:
        target = str(row.get("target") or "")
        if target_counts[target] >= 2:
            continue
        image = row.get("image") or ""
        if image in used_images:
            continue
        selected.append(row)
        used_images.add(image)
        target_counts[target] += 1
        if len(selected) >= count:
            return selected

    # Second pass: fill from the remaining manifest order.
    for row in rows:
        image = row.get("image") or ""
        if image in used_images:
            continue
        selected.append(row)
        used_images.add(image)
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
    if not args.gold_manifest.exists():
        raise FileNotFoundError(f"Gold manifest not found: {args.gold_manifest}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = args.output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    for old in images_dir.glob("ANCH*.jpg"):
        old.unlink()

    manifest = _read_manifest(args.gold_manifest)
    selected = _select_rows(manifest, args.count)
    template_rows: list[dict[str, object]] = []

    rendered = 0
    for i, row in enumerate(selected, start=1):
        image_path = Path(row["image"])
        img = cv2.imread(str(image_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        shown, scale = _resize_to_width(img, args.max_image_width)
        out_name = f"ANCH{i:03d}_{row['sample_id']}_T{row['target']}_{image_path.name}"
        out_path = images_dir / out_name
        _draw_text(
            shown,
            [
                f"ANCH{i:03d} source={row['sample_id']} Target{row['target']} image={image_path.name}",
                "Fill anchors.csv with original image pixel coordinates, not resized display pixels.",
                "railing: bottom and top of the same vertical post, true_height_m=0.92",
                "Ignore curb anchors for this dataset; use railing height only.",
            ],
        )
        cv2.imwrite(str(out_path), shown)
        rendered += 1

        template_rows.append(
            {
                "anchor_id": f"ANCH{i:03d}_railing",
                "source_sample_id": row["sample_id"],
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
                "notes": "",
            }
        )

    _write_csv(args.output_dir / "anchors_template.csv", template_rows)
    # Keep a working copy that the user can edit.
    _write_csv(args.output_dir / "anchors.csv", template_rows)

    print("Anchor review package prepared.")
    print(f"Review images: {images_dir} ({rendered}/{len(selected)} rendered)")
    print(f"Fill this file: {args.output_dir / 'anchors.csv'}")
    print()
    print("For each usable image, fill the railing anchor row:")
    print("- railing: bottom_u,bottom_v and top_u,top_v on the same vertical railing post; true_height_m=0.92")
    print("Use original image pixel coordinates. Leave unclear rows blank.")


if __name__ == "__main__":
    main()
