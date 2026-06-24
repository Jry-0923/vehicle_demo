import argparse
import json
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize exported vehicle centers on images")
    p.add_argument("--centers-json", type=Path, required=True, help="Path to export_vehicle_centers JSON")
    p.add_argument("--output-dir", type=Path, required=True, help="Directory to save rendered frames")
    p.add_argument("--limit", type=int, default=0, help="Optional max number of frames to render")
    p.add_argument("--only-ok", action="store_true", help="Render only frames whose status starts with 'ok'")
    p.add_argument("--point-radius", type=int, default=6)
    return p.parse_args()


def draw_point(img, uv, color, text, r=6):
    if uv is None or len(uv) != 2:
        return
    u, v = int(round(float(uv[0]))), int(round(float(uv[1])))
    h, w = img.shape[:2]
    if not (0 <= u < w and 0 <= v < h):
        return
    cv2.circle(img, (u, v), r, color, -1)
    cv2.circle(img, (u, v), r + 2, (255, 255, 255), 1)
    cv2.putText(img, text, (u + 8, v - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def color_for_idx(i: int):
    palette = [
        (0, 200, 0),
        (255, 120, 0),
        (255, 0, 255),
        (0, 180, 255),
        (200, 0, 0),
        (0, 255, 180),
    ]
    return palette[i % len(palette)]


def main() -> None:
    args = parse_args()
    if not args.centers_json.exists():
        raise FileNotFoundError(f"JSON not found: {args.centers_json}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.centers_json.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    frames = obj.get("frames", [])
    if args.only_ok:
        frames = [x for x in frames if str(x.get("status", "")).startswith("ok")]
    if args.limit > 0:
        frames = frames[: args.limit]

    written = 0
    skipped = 0
    for i, fr in enumerate(frames, start=1):
        img_path = fr.get("image")
        if not img_path:
            skipped += 1
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            skipped += 1
            continue

        status = str(fr.get("status", "unknown"))
        mode = str(fr.get("status_temporal") or fr.get("selected_mode") or fr.get("selected_label_source") or "")
        detections = fr.get("filtered_detections") if "filtered_detections" in fr else (fr.get("detections") or [])
        if detections:
            for di, det in enumerate(detections):
                c = color_for_idx(di)
                box = det.get("box_xyxy")
                if box and len(box) == 4:
                    x1, y1, x2, y2 = [int(v) for v in box]
                    cv2.rectangle(img, (x1, y1), (x2, y2), c, 2)
                    label = det.get("label") or "veh"
                    rv = det.get("radar_validation") or {}
                    m = int(det.get("matched_points", rv.get("matched_points", 0)) or 0)
                    cv2.putText(img, f"{di}:{label} m={m}", (x1, max(16, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
                draw_point(img, det.get("radar_surface_uv"), (0, 0, 255), f"s{di}", r=args.point_radius)
                draw_point(
                    img,
                    det.get("estimated_center_uv") or det.get("pseudo_gt_center_uv"),
                    (0, 255, 255),
                    f"c{di}",
                    r=args.point_radius,
                )
                draw_point(img, det.get("geometry_projected_uv"), (255, 0, 255), f"p{di}", r=max(3, args.point_radius - 2))
                draw_point(img, det.get("estimated_head_uv"), (255, 180, 0), f"h{di}", r=args.point_radius)
        else:
            box = fr.get("selected_box_xyxy")
            if box and len(box) == 4:
                x1, y1, x2, y2 = [int(v) for v in box]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 2)
            draw_point(img, fr.get("radar_surface_uv"), (0, 0, 255), "surface", r=args.point_radius)
            draw_point(img, fr.get("estimated_center_uv") or fr.get("pseudo_gt_center_uv"), (0, 255, 255), "center", r=args.point_radius)
            draw_point(img, fr.get("estimated_head_uv"), (255, 180, 0), "head", r=args.point_radius)

        rv = fr.get("radar_validation") or {}
        lines = [
            f"idx={fr.get('frame_index', i)} status={status} mode={mode}",
            f"img_ts={fr.get('image_timestamp', 'NA')} radar_ts={fr.get('radar_timestamp', 'NA')}",
            f"detections={len(detections)} matched_points={fr.get('matched_points', rv.get('matched_points', 0))} matched_ids={fr.get('matched_ids', None)}",
        ]
        y = 24
        for line in lines:
            cv2.putText(img, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 0), 2)
            y += 26

        out_name = f"{i:05d}_{Path(img_path).name}"
        out_path = args.output_dir / out_name
        cv2.imwrite(str(out_path), img)
        written += 1

    print(f"Rendered {written} frames to {args.output_dir} (skipped={skipped})")


if __name__ == "__main__":
    main()
