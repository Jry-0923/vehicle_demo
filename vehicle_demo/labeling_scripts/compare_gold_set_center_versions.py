#!/usr/bin/env python3
"""Compare original, anchor-scaled, and optional box-completed centers on the gold set."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


FIELDS = [
    "sample_id",
    "target",
    "image_name",
    "manual_grade",
    "center_error_type",
    "original_center_m",
    "original_depth_z_m",
    "scale_corrected_center_m",
    "scale_corrected_depth_z_m",
    "scale_shift_m",
    "box_completed_center_m",
    "box_completed_depth_z_m",
    "completed_shift_from_original_m",
    "notes",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare center versions for the gold-set samples.")
    p.add_argument("--gold-manifest", type=Path, default=Path("outputs/gold_set_review/manifest.csv"))
    p.add_argument("--annotations-csv", type=Path, default=Path("outputs/gold_set_review/annotations.csv"))
    p.add_argument("--scaled-dir", type=Path, default=Path("outputs/anchor_scaled_labels"))
    p.add_argument("--completed-json", type=Path, default=None, help="Optional JSON containing completed box centers.")
    p.add_argument("--output-csv", type=Path, default=Path("outputs/anchor_scale_review/gold_set_center_comparison.csv"))
    p.add_argument("--summary-json", type=Path, default=Path("outputs/anchor_scale_review/gold_set_center_comparison_summary.json"))
    return p.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _point(v: Any) -> np.ndarray | None:
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except Exception:
            return None
    if not isinstance(v, list) or len(v) < 3:
        return None
    try:
        arr = np.asarray([float(v[0]), float(v[1]), float(v[2])], dtype=np.float32)
    except Exception:
        return None
    return arr if np.isfinite(arr).all() else None


def _image_name(path: str | None) -> str:
    return Path(path or "").name


def _index_scaled(scaled_dir: Path) -> dict[tuple[int, str, int], dict[str, Any]]:
    out: dict[tuple[int, str, int], dict[str, Any]] = {}
    for target in range(1, 8):
        path = scaled_dir / f"4_15_target{target}_final_labels_anchor_scaled.json"
        if not path.exists():
            continue
        obj = json.load(path.open("r", encoding="utf-8"))
        for frame in obj.get("frames", []):
            image = _image_name(frame.get("image"))
            for det_i, det in enumerate(frame.get("final_detections") or []):
                out[(target, image, det_i)] = det
    return out


def _index_completed(path: Path | None) -> dict[tuple[int, str, int], dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    obj = json.load(path.open("r", encoding="utf-8"))
    out: dict[tuple[int, str, int], dict[str, Any]] = {}
    for det_i, det in enumerate(obj.get("detections") or []):
        target = det.get("target")
        image = _image_name(det.get("image"))
        source_i = det.get("source_det_index", det.get("det_index", det_i))
        if target is None or not image:
            continue
        out[(int(target), image, int(source_i))] = det
    return out


def _fmt(pt: np.ndarray | None) -> str:
    if pt is None:
        return ""
    return json.dumps([float(x) for x in pt.tolist()])


def main() -> None:
    args = parse_args()
    manifest = _read_csv(args.gold_manifest)
    annotations = {r["sample_id"]: r for r in _read_csv(args.annotations_csv)}
    scaled = _index_scaled(args.scaled_dir)
    completed = _index_completed(args.completed_json)

    rows = []
    for row in manifest:
        sample_id = row["sample_id"]
        target = int(row["target"])
        image_name = row["image_name"]
        det_i = int(row.get("det_index") or 0)
        ann = annotations.get(sample_id, {})
        orig = _point(row.get("center_m"))
        scaled_det = scaled.get((target, image_name, det_i), {})
        scaled_center = _point(scaled_det.get("scale_corrected_final_vehicle_center_m"))
        if scaled_center is None:
            scaled_center = _point(scaled_det.get("scale_corrected_smoothed_pseudo_gt_vehicle_center_m"))
        comp_det = completed.get((target, image_name, det_i), {})
        comp_center = _point(
            (comp_det.get("completion") or {}).get("aligned_completed_box_center_m")
            or (comp_det.get("completion") or {}).get("completed_box_center_m")
            or comp_det.get("box_completed_center_m")
        )
        scale_shift = float(np.linalg.norm(scaled_center - orig)) if orig is not None and scaled_center is not None else None
        comp_shift = float(np.linalg.norm(comp_center - orig)) if orig is not None and comp_center is not None else None
        rows.append(
            {
                "sample_id": sample_id,
                "target": target,
                "image_name": image_name,
                "manual_grade": ann.get("manual_grade", ""),
                "center_error_type": ann.get("center_error_type", ""),
                "original_center_m": _fmt(orig),
                "original_depth_z_m": "" if orig is None else float(orig[2]),
                "scale_corrected_center_m": _fmt(scaled_center),
                "scale_corrected_depth_z_m": "" if scaled_center is None else float(scaled_center[2]),
                "scale_shift_m": "" if scale_shift is None else scale_shift,
                "box_completed_center_m": _fmt(comp_center),
                "box_completed_depth_z_m": "" if comp_center is None else float(comp_center[2]),
                "completed_shift_from_original_m": "" if comp_shift is None else comp_shift,
                "notes": ann.get("manual_notes", ""),
            }
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    scale_shifts = [float(r["scale_shift_m"]) for r in rows if r["scale_shift_m"] != ""]
    comp_shifts = [float(r["completed_shift_from_original_m"]) for r in rows if r["completed_shift_from_original_m"] != ""]
    summary = {
        "task": "gold_set_center_version_comparison",
        "rows": len(rows),
        "manual_grade_counts": dict(Counter(r["manual_grade"] for r in rows)),
        "center_error_type_counts": dict(Counter(r["center_error_type"] for r in rows)),
        "scale_corrected_available": len(scale_shifts),
        "scale_shift_m_mean": float(np.mean(scale_shifts)) if scale_shifts else None,
        "scale_shift_m_median": float(np.median(scale_shifts)) if scale_shifts else None,
        "box_completed_available": len(comp_shifts),
        "completed_shift_m_mean": float(np.mean(comp_shifts)) if comp_shifts else None,
        "completed_shift_m_median": float(np.median(comp_shifts)) if comp_shifts else None,
        "output_csv": str(args.output_csv),
    }
    with args.summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
