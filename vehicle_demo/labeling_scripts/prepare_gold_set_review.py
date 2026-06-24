#!/usr/bin/env python3
"""Prepare a small human review gold set for 4.15 vehicle-center labels."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ANNOTATION_COLUMNS = [
    "sample_id",
    "manual_grade",
    "view_type",
    "occlusion",
    "center_error_type",
    "manual_ground_contact_uv",
    "manual_heading_uv",
    "manual_width_uv",
    "manual_notes",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Select a stratified 4.15 gold-set sample and render review images. "
            "Manual labels are written into annotations.csv."
        )
    )
    p.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    p.add_argument(
        "--labels-dir",
        type=Path,
        default=Path("outputs/monodetr_upgraded_50m_all_strong_only"),
        help="Directory containing 4_15_targetN_final_labels_monodetr_upgraded.json.",
    )
    p.add_argument("--output-dir", type=Path, default=Path("outputs/gold_set_review"))
    p.add_argument("--count", type=int, default=40)
    p.add_argument("--targets", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7])
    p.add_argument("--seed", type=int, default=415)
    p.add_argument("--max-image-width", type=int, default=1600)
    return p.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _image_name(path: str | None) -> str:
    return Path(path or "").name


def _point3(v: Any) -> list[float] | None:
    if not isinstance(v, list) or len(v) < 3:
        return None
    try:
        out = [float(v[0]), float(v[1]), float(v[2])]
    except Exception:
        return None
    return out if np.isfinite(out).all() else None


def _point2(v: Any) -> list[float] | None:
    if not isinstance(v, list) or len(v) < 2:
        return None
    try:
        out = [float(v[0]), float(v[1])]
    except Exception:
        return None
    return out if np.isfinite(out).all() else None


def _box(v: Any) -> list[float] | None:
    if not isinstance(v, list) or len(v) != 4:
        return None
    try:
        out = [float(x) for x in v]
    except Exception:
        return None
    if not np.isfinite(out).all() or out[2] <= out[0] or out[3] <= out[1]:
        return None
    return out


def _depth_bin(z: float) -> str:
    if z < 10.0:
        return "near_lt10m"
    if z <= 20.0:
        return "mid_10_20m"
    return "far_gt20m"


def _edge_touch(box: list[float], width: int, height: int, margin: int = 8) -> bool:
    x1, y1, x2, y2 = box
    return x1 <= margin or y1 <= margin or x2 >= width - margin or y2 >= height - margin


def _monodetr_status(det: dict[str, Any]) -> tuple[str, float | None, float | None]:
    val = det.get("monodetr_validation") or {}
    if val.get("strong_validation"):
        metrics = val.get("metrics") or {}
        return "strong_agree", metrics.get("best_iou"), metrics.get("depth_delta_m")
    metrics = val.get("metrics") or {}
    best_iou = metrics.get("best_iou")
    depth_delta = metrics.get("depth_delta_m")
    if val.get("matched_detection") is None:
        return "no_monodetr_car_box", best_iou, depth_delta
    try:
        if best_iou is not None and float(best_iou) < 0.5:
            return "monodetr_iou_lt_0_5", best_iou, depth_delta
        if depth_delta is not None and float(depth_delta) > 5.0:
            return "monodetr_depth_gt_5m", best_iou, depth_delta
    except Exception:
        pass
    return str(val.get("status") or "monodetr_not_sampled"), best_iou, depth_delta


def _load_candidates(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    project_root = args.outputs_dir.resolve().parent
    for target in args.targets:
        path = args.labels_dir / f"4_15_target{target}_final_labels_monodetr_upgraded.json"
        if not path.exists():
            path = args.outputs_dir / f"4_15_target{target}_final_labels_baseline.json"
        obj = _load_json(path)
        for frame in obj.get("frames", []):
            image_path = frame.get("image")
            if image_path:
                img_p = Path(str(image_path))
                if not img_p.is_absolute():
                    image_path = str(project_root / img_p)
            image_name = _image_name(image_path)
            final_dets = frame.get("final_detections") or []
            for det_idx, det in enumerate(final_dets):
                if not det.get("use_for_training", True):
                    continue
                center = _point3(
                    det.get("final_vehicle_center_m")
                    or det.get("smoothed_pseudo_gt_vehicle_center_m")
                    or det.get("pseudo_gt_vehicle_center_m")
                )
                box = _box(det.get("box_xyxy"))
                if center is None or box is None:
                    continue
                md_status, md_iou, md_depth_delta = _monodetr_status(det)
                box_w, box_h = box[2] - box[0], box[3] - box[1]
                rows.append(
                    {
                        "target": int(target),
                        "frame_index": int(frame.get("frame_index") or 0),
                        "image": str(image_path),
                        "image_name": image_name,
                        "det_index": int(det_idx),
                        "num_final_detections": int(len(final_dets)),
                        "box_xyxy": box,
                        "box_area_px": float(box_w * box_h),
                        "center_m": center,
                        "center_z_m": float(center[2]),
                        "depth_bin": _depth_bin(float(center[2])),
                        "label": det.get("label") or frame.get("selected_label") or "vehicle",
                        "label_confidence": det.get("label_confidence"),
                        "final_label_grade": det.get("final_label_grade") or frame.get("final_label_grade"),
                        "final_label_reason": det.get("final_label_reason"),
                        "pred_center_uv": _point2(det.get("geometry_projected_uv") or det.get("pseudo_gt_center_uv")),
                        "pseudo_gt_center_uv": _point2(det.get("pseudo_gt_center_uv")),
                        "monodetr_status": md_status,
                        "monodetr_best_iou": md_iou,
                        "monodetr_depth_delta_m": md_depth_delta,
                        "monodetr_box_xyxy": _box(((det.get("monodetr_validation") or {}).get("matched_detection") or {}).get("bbox_2d")),
                        "low_confidence_reasons": ";".join(det.get("low_confidence_reasons") or []),
                    }
                )
    return rows


def _image_size(path: str) -> tuple[int, int] | None:
    img = cv2.imread(path)
    if img is None:
        return None
    h, w = img.shape[:2]
    return w, h


def _decorate_candidates(rows: list[dict[str, Any]]) -> None:
    size_cache: dict[str, tuple[int, int] | None] = {}
    for r in rows:
        size_cache.setdefault(r["image"], _image_size(r["image"]))
        size = size_cache[r["image"]]
        r["image_width"] = None if size is None else size[0]
        r["image_height"] = None if size is None else size[1]
        r["edge_touch"] = False
        if size is not None:
            r["edge_touch"] = _edge_touch(r["box_xyxy"], size[0], size[1])
        r["multi_vehicle"] = int(r["num_final_detections"]) >= 2
        r["large_box"] = float(r["box_area_px"]) >= 60000.0


def _stable_sort(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    keyed = [(float(rng.random()), r) for r in rows]
    keyed.sort(key=lambda x: (x[0], x[1]["target"], x[1]["frame_index"], x[1]["det_index"]))
    return [r for _, r in keyed]


def _select(rows: list[dict[str, Any]], count: int, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = [r for r in rows if r.get("image_width") is not None]
    shuffled = _stable_sort(rows, seed)
    selected: list[dict[str, Any]] = []
    used_images: set[str] = set()

    def add_from(name: str, pred, quota: int) -> int:
        added = 0
        for r in shuffled:
            if added >= quota or len(selected) >= count:
                break
            key = r["image"]
            if key in used_images or not pred(r):
                continue
            r = dict(r)
            r["selection_bucket"] = name
            selected.append(r)
            used_images.add(key)
            added += 1
        return added

    bucket_counts: dict[str, int] = {}

    target_min = max(1, min(3, count // 12))
    for target in sorted({int(r["target"]) for r in rows}):
        current = sum(1 for r in selected if int(r["target"]) == target)
        need = max(0, target_min - current)
        if need:
            bucket_counts[f"target{target}_coverage"] = add_from(
                f"target{target}_coverage", lambda r, t=target: int(r["target"]) == t, need
            )

    quotas = [
        ("monodetr_no_car_box", lambda r: r["monodetr_status"] == "no_monodetr_car_box", 4),
        ("monodetr_low_iou", lambda r: r["monodetr_status"] == "monodetr_iou_lt_0_5", 3),
        ("monodetr_depth_gt_5m", lambda r: r["monodetr_status"] == "monodetr_depth_gt_5m", 4),
        ("far_gt20m", lambda r: r["depth_bin"] == "far_gt20m", 4),
        ("near_lt10m", lambda r: r["depth_bin"] == "near_lt10m", 4),
        ("mid_10_20m", lambda r: r["depth_bin"] == "mid_10_20m", 4),
        ("multi_vehicle", lambda r: bool(r["multi_vehicle"]), 2),
        ("edge_or_large_box", lambda r: bool(r["edge_touch"] or r["large_box"]), 2),
        ("monodetr_strong_agree", lambda r: r["monodetr_status"] == "strong_agree", 3),
    ]
    for name, pred, quota in quotas:
        if len(selected) >= count:
            bucket_counts[name] = 0
            continue
        bucket_counts[name] = add_from(name, pred, min(quota, count - len(selected)))

    add_from("balanced_fill", lambda r: True, count - len(selected))
    selected = selected[:count]
    for i, r in enumerate(selected, start=1):
        r["sample_id"] = f"GS{i:03d}"

    summary = {
        "requested_count": int(count),
        "selected_count": int(len(selected)),
        "target_counts": dict(sorted(Counter(r["target"] for r in selected).items())),
        "depth_bin_counts": dict(sorted(Counter(r["depth_bin"] for r in selected).items())),
        "monodetr_status_counts": dict(sorted(Counter(r["monodetr_status"] for r in selected).items())),
        "selection_bucket_counts": dict(sorted(Counter(r["selection_bucket"] for r in selected).items())),
        "candidate_count": int(len(rows)),
        "bucket_fill_attempts": bucket_counts,
    }
    return selected, summary


def _scale_to_width(img: np.ndarray, max_width: int) -> tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    if max_width <= 0 or w <= max_width:
        return img, 1.0
    scale = float(max_width) / float(w)
    resized = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    return resized, scale


def _draw_box(img: np.ndarray, box: list[float] | None, color: tuple[int, int, int], label: str, scale: float = 1.0) -> None:
    if box is None:
        return
    x1, y1, x2, y2 = [int(round(float(v) * scale)) for v in box]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
    cv2.putText(img, label, (x1, max(24, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2)


def _draw_point(img: np.ndarray, uv: list[float] | None, color: tuple[int, int, int], label: str, scale: float = 1.0) -> None:
    if uv is None:
        return
    u, v = int(round(float(uv[0]) * scale)), int(round(float(uv[1]) * scale))
    h, w = img.shape[:2]
    if not (0 <= u < w and 0 <= v < h):
        return
    cv2.circle(img, (u, v), 8, color, -1)
    cv2.circle(img, (u, v), 11, (255, 255, 255), 2)
    cv2.putText(img, label, (u + 12, max(22, v - 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2)


def _wrap_lines(text: str, max_chars: int = 82) -> list[str]:
    words = text.split()
    lines: list[str] = []
    cur: list[str] = []
    for word in words:
        if sum(len(x) + 1 for x in cur) + len(word) > max_chars and cur:
            lines.append(" ".join(cur))
            cur = [word]
        else:
            cur.append(word)
    if cur:
        lines.append(" ".join(cur))
    return lines


def _render_review_image(row: dict[str, Any], out_path: Path, max_width: int) -> bool:
    img = cv2.imread(row["image"])
    if img is None:
        return False
    img, scale = _scale_to_width(img, max_width)
    _draw_box(img, row["box_xyxy"], (0, 220, 0), f"{row['sample_id']} TARGET {row['det_index']}", scale)
    _draw_box(img, row.get("monodetr_box_xyxy"), (255, 80, 0), "MonoDETR", scale)
    _draw_point(img, row.get("pred_center_uv"), (0, 255, 255), "projected center", scale)
    _draw_point(img, row.get("pseudo_gt_center_uv"), (255, 0, 255), "pointmap median uv", scale)

    panel_lines = [
        f"{row['sample_id']}  Target{row['target']} frame={row['frame_index']} det={row['det_index']} image={row['image_name']}",
        f"center_m=[{row['center_m'][0]:.2f}, {row['center_m'][1]:.2f}, {row['center_m'][2]:.2f}] depth_bin={row['depth_bin']}",
        f"grade={row.get('final_label_grade')} reason={row.get('final_label_reason')} conf={row.get('label_confidence')}",
        f"MonoDETR={row['monodetr_status']} IoU={row.get('monodetr_best_iou')} depth_delta={row.get('monodetr_depth_delta_m')}",
        f"multi_vehicle={row['multi_vehicle']} edge_touch={row['edge_touch']} large_box={row['large_box']} bucket={row['selection_bucket']}",
        "Annotate: manual_grade A/B/C; view_type front/rear/side/oblique; occlusion none/partial/heavy;",
        "center_error_type ok/surface_biased/too_near/too_far/wrong_vehicle/bad_2d_box/occluded_unusable.",
    ]
    y = 28
    for line in panel_lines:
        for wrapped in _wrap_lines(line):
            cv2.putText(img, wrapped, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 4)
            cv2.putText(img, wrapped, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
            y += 24
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(out_path), img))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = args.output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    for old_image in images_dir.glob("GS*.jpg"):
        old_image.unlink()
    rows = _load_candidates(args)
    _decorate_candidates(rows)
    selected, summary = _select(rows, args.count, args.seed)

    manifest_rows: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    rendered = 0
    for row in selected:
        image_out = images_dir / f"{row['sample_id']}_T{row['target']}_{Path(row['image']).name}"
        if _render_review_image(row, image_out, args.max_image_width):
            rendered += 1
        manifest = {
            **row,
            "review_image": str(image_out),
            "box_xyxy": json.dumps(row["box_xyxy"], ensure_ascii=False),
            "center_m": json.dumps(row["center_m"], ensure_ascii=False),
            "pred_center_uv": json.dumps(row.get("pred_center_uv"), ensure_ascii=False),
            "pseudo_gt_center_uv": json.dumps(row.get("pseudo_gt_center_uv"), ensure_ascii=False),
            "monodetr_box_xyxy": json.dumps(row.get("monodetr_box_xyxy"), ensure_ascii=False),
        }
        manifest_rows.append(manifest)
        annotations.append(
            {
                "sample_id": row["sample_id"],
                "manual_grade": "",
                "view_type": "",
                "occlusion": "",
                "center_error_type": "",
                "manual_ground_contact_uv": "",
                "manual_heading_uv": "",
                "manual_width_uv": "",
                "manual_notes": "",
            }
        )

    manifest_fields = [
        "sample_id",
        "target",
        "frame_index",
        "image_name",
        "image",
        "review_image",
        "det_index",
        "num_final_detections",
        "box_xyxy",
        "center_m",
        "center_z_m",
        "depth_bin",
        "pred_center_uv",
        "pseudo_gt_center_uv",
        "final_label_grade",
        "final_label_reason",
        "label_confidence",
        "monodetr_status",
        "monodetr_best_iou",
        "monodetr_depth_delta_m",
        "monodetr_box_xyxy",
        "multi_vehicle",
        "edge_touch",
        "large_box",
        "selection_bucket",
        "low_confidence_reasons",
    ]
    _write_csv(args.output_dir / "manifest.csv", manifest_rows, manifest_fields)
    _write_csv(args.output_dir / "annotations.csv", annotations, ANNOTATION_COLUMNS)

    instructions = {
        "task": "Human review gold set for vehicle 3D center labels",
        "what_to_open": str(images_dir),
        "what_to_fill": str(args.output_dir / "annotations.csv"),
        "fields": {
            "manual_grade": "A if projected center and BEV/depth look like actual vehicle box center, B if usable but biased, C if wrong/unusable.",
            "view_type": "front, rear, side, or oblique.",
            "occlusion": "none, partial, or heavy.",
            "center_error_type": "ok, surface_biased, too_near, too_far, wrong_vehicle, bad_2d_box, or occluded_unusable.",
            "manual_ground_contact_uv": "Optional pixel [u,v] for approximate vehicle bottom center on ground.",
            "manual_heading_uv": "Optional two pixels [[u1,v1],[u2,v2]] showing vehicle forward direction.",
            "manual_width_uv": "Optional two pixels [[u_left,v_left],[u_right,v_right]] across vehicle width.",
            "manual_notes": "Short reason for B/C or any ambiguity.",
        },
        "overlay_legend": {
            "green_box": "target pseudo-label vehicle box",
            "cyan_dot": "3D center projected back into the image",
            "magenta_dot": "median point location inside the MASt3R pointmap box",
            "blue_box": "MonoDETR matched box when available",
        },
        "summary": summary,
    }
    with (args.output_dir / "instructions.json").open("w", encoding="utf-8") as f:
        json.dump(instructions, f, ensure_ascii=False, indent=2)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Gold-set review package prepared.")
    print(f"Review images: {images_dir} ({rendered}/{len(selected)} rendered)")
    print(f"Manifest: {args.output_dir / 'manifest.csv'}")
    print(f"Fill this file: {args.output_dir / 'annotations.csv'}")
    print()
    print("For each review image, annotate these fields:")
    for col in ANNOTATION_COLUMNS[1:]:
        print(f"- {col}")
    print()
    print("Allowed core values:")
    print("- manual_grade: A, B, C")
    print("- view_type: front, rear, side, oblique")
    print("- occlusion: none, partial, heavy")
    print("- center_error_type: ok, surface_biased, too_near, too_far, wrong_vehicle, bad_2d_box, occluded_unusable")
    print()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
