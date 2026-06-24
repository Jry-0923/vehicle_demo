#!/usr/bin/env python3
"""Click bottom/top image points for fixed-height anchor rows."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2


POINT_FIELDS = ("bottom_u", "bottom_v", "top_u", "top_v")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Interactively fill anchor endpoint pixels in anchors.csv.")
    p.add_argument("--anchors-csv", type=Path, default=Path("outputs/anchor_scale_review/anchors.csv"))
    p.add_argument("--max-window-width", type=int, default=1500)
    p.add_argument("--max-window-height", type=int, default=900)
    p.add_argument("--anchor-type", choices=["railing", "curb"], default=None)
    p.add_argument("--start-anchor-id", default=None)
    p.add_argument("--overwrite", action="store_true", help="Relabel rows that already have complete endpoints.")
    return p.parse_args()


def _read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def _write_rows(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def _complete(row: dict[str, str]) -> bool:
    return all(str(row.get(k, "")).strip() for k in POINT_FIELDS)


def _scale_for_window(width: int, height: int, max_width: int, max_height: int) -> float:
    scale = min(float(max_width) / float(width), float(max_height) / float(height), 1.0)
    return max(scale, 0.05)


def _draw_overlay(display, row: dict[str, str], points: list[tuple[int, int]], scale: float) -> None:
    anchor_id = row.get("anchor_id", "")
    anchor_type = row.get("anchor_type", "")
    height = row.get("true_height_m", "")
    image_name = row.get("image_name", Path(row.get("image", "")).name)
    lines = [
        f"{anchor_id}  {anchor_type}  true_height={height}m  image={image_name}",
        "Left click: bottom point, then top point. Enter/Space: save. u: undo. s: skip. q: quit.",
        "bottom = lower endpoint in the image; top = upper endpoint of the same physical vertical height.",
    ]
    y = 26
    for line in lines:
        cv2.putText(display, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4)
        cv2.putText(display, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
        y += 27

    colors = [(0, 255, 0), (0, 0, 255)]
    labels = ["bottom", "top"]
    for i, (u, v) in enumerate(points):
        x = int(round(u * scale))
        y = int(round(v * scale))
        color = colors[min(i, 1)]
        cv2.circle(display, (x, y), 7, color, -1)
        cv2.circle(display, (x, y), 10, (0, 0, 0), 2)
        cv2.putText(display, labels[i], (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4)
        cv2.putText(display, labels[i], (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    if len(points) == 2:
        p0 = (int(round(points[0][0] * scale)), int(round(points[0][1] * scale)))
        p1 = (int(round(points[1][0] * scale)), int(round(points[1][1] * scale)))
        cv2.line(display, p0, p1, (255, 255, 255), 5)
        cv2.line(display, p0, p1, (0, 200, 255), 2)


def _label_row(row: dict[str, str], max_width: int, max_height: int) -> tuple[str, list[tuple[int, int]]]:
    image_path = Path(row["image"])
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"skip missing image: {image_path}")
        return "skip", []

    height, width = img.shape[:2]
    scale = _scale_for_window(width, height, max_width, max_height)
    display_size = (int(round(width * scale)), int(round(height * scale)))
    window = "anchor_labeler"
    points: list[tuple[int, int]] = []

    def on_mouse(event, x, y, _flags, _param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN or len(points) >= 2:
            return
        u = int(round(x / scale))
        v = int(round(y / scale))
        u = min(max(u, 0), width - 1)
        v = min(max(v, 0), height - 1)
        points.append((u, v))

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    print(
        f"label {row.get('anchor_id')}: click bottom then top in the image window. "
        "Enter/Space=save, u=undo, s=skip, q/Esc/window close=quit."
    )

    while True:
        if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            return "quit", points
        display = cv2.resize(img, display_size, interpolation=cv2.INTER_AREA) if scale != 1.0 else img.copy()
        _draw_overlay(display, row, points, scale)
        cv2.imshow(window, display)
        key = cv2.waitKey(30) & 0xFF
        if key in (13, 10, 32):
            if len(points) == 2:
                return "save", points
            print("Need two clicks before saving.")
        elif key == ord("u"):
            if points:
                points.pop()
        elif key == ord("s"):
            return "skip", points
        elif key in (ord("q"), ord("Q"), 27):
            return "quit", points


def main() -> None:
    args = parse_args()
    if not args.anchors_csv.exists():
        raise FileNotFoundError(f"Anchors CSV not found: {args.anchors_csv}")

    fields, rows = _read_rows(args.anchors_csv)
    missing = [f for f in POINT_FIELDS if f not in fields]
    if missing:
        raise ValueError(f"Missing required columns in {args.anchors_csv}: {missing}")

    start_seen = args.start_anchor_id is None
    saved = 0
    skipped_complete = 0
    visited = 0

    try:
        for row in rows:
            if args.anchor_type and row.get("anchor_type") != args.anchor_type:
                continue
            if not start_seen:
                start_seen = row.get("anchor_id") == args.start_anchor_id
                if not start_seen:
                    continue
            if _complete(row) and not args.overwrite:
                skipped_complete += 1
                continue

            visited += 1
            action, points = _label_row(row, args.max_window_width, args.max_window_height)
            if action == "save":
                (bottom_u, bottom_v), (top_u, top_v) = points
                row["bottom_u"] = str(bottom_u)
                row["bottom_v"] = str(bottom_v)
                row["top_u"] = str(top_u)
                row["top_v"] = str(top_v)
                saved += 1
                _write_rows(args.anchors_csv, fields, rows)
                print(f"saved {row.get('anchor_id')}: bottom=({bottom_u},{bottom_v}) top=({top_u},{top_v})")
            elif action == "quit":
                break
    except KeyboardInterrupt:
        print("\ninterrupted; saving current CSV before exit.")
    finally:
        cv2.destroyAllWindows()
        _write_rows(args.anchors_csv, fields, rows)

    print(f"done: visited={visited}, saved={saved}, skipped_complete={skipped_complete}, csv={args.anchors_csv}")


if __name__ == "__main__":
    main()
