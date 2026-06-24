#!/usr/bin/env python3
"""Upgrade image pseudo-GT label grades with MonoDETR as a validation signal.

This script does not replace final_vehicle_center_m. MonoDETR is only used to
validate the existing image-dominant pseudo label.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Upgrade A/B/C label grades using MonoDETR validation.")
    p.add_argument("--monodetr-json", type=Path, required=True, help="Output from export_monodetr_smoke.py")
    p.add_argument("--baseline-dir", type=Path, default=Path("outputs"))
    p.add_argument("--targets", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7])
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--summary-json", type=Path, required=True)
    p.add_argument("--min-iou", type=float, default=0.5)
    p.add_argument("--max-depth-delta-m", type=float, default=5.0)
    p.add_argument("--required-class", type=str, default="Car")
    p.add_argument(
        "--mark-depth-conflicts-c",
        action="store_true",
        help="Mark labels C when a matching MonoDETR car box exists but depth disagrees.",
    )
    return p.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _image_name(path: str | None) -> str:
    return Path(path or "").name


def _box(v: Any) -> list[float] | None:
    if not isinstance(v, list) or len(v) != 4:
        return None
    try:
        box = [float(x) for x in v]
    except Exception:
        return None
    if not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
        return None
    return box


def _center(v: Any) -> list[float] | None:
    if not isinstance(v, list) or len(v) < 3:
        return None
    try:
        c = [float(v[0]), float(v[1]), float(v[2])]
    except Exception:
        return None
    return c if np.isfinite(c).all() else None


def _iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


def _grade_rank(grade: str | None) -> int:
    return {"A": 3, "B": 2, "C": 1}.get(str(grade or ""), 0)


def _index_monodetr(monodetr: dict[str, Any]) -> dict[tuple[int, str], dict[str, Any]]:
    indexed: dict[tuple[int, str], dict[str, Any]] = {}
    for row in monodetr.get("evaluation", []):
        target = row.get("target")
        image = row.get("image_name") or _image_name(row.get("image"))
        if target is None or not image:
            continue
        indexed[(int(target), image)] = row
    return indexed


def _best_monodetr_match(
    detections: list[dict[str, Any]],
    gt_box: list[float],
    gt_center: list[float] | None,
    required_class: str,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    best: dict[str, Any] | None = None
    best_iou = 0.0
    for det in detections:
        if det.get("class") != required_class:
            continue
        box = _box(det.get("bbox_2d"))
        if box is None:
            continue
        iou = _iou(gt_box, box)
        if iou > best_iou:
            best = det
            best_iou = iou

    depth_delta = None
    if best is not None and gt_center is not None:
        depth = best.get("depth")
        if isinstance(depth, (int, float)):
            depth_delta = abs(float(depth) - float(gt_center[2]))

    return best, {
        "best_iou": float(best_iou),
        "depth_delta_m": depth_delta,
        "required_class": required_class,
    }


def _validate_detection(det: dict[str, Any], md_row: dict[str, Any] | None, args: argparse.Namespace) -> tuple[str, dict[str, Any], bool]:
    gt_box = _box(det.get("box_xyxy"))
    gt_center = _center(
        det.get("final_vehicle_center_m")
        or det.get("smoothed_pseudo_gt_vehicle_center_m")
        or det.get("pseudo_gt_vehicle_center_m")
    )
    previous_grade = str(det.get("final_label_grade") or "B")

    if md_row is None:
        return previous_grade, {"status": "not_sampled_by_monodetr"}, bool(det.get("use_for_training", True))
    if gt_box is None:
        return previous_grade, {"status": "missing_gt_box"}, bool(det.get("use_for_training", True))

    md_det, metrics = _best_monodetr_match(md_row.get("detections") or [], gt_box, gt_center, args.required_class)
    validation = {
        "status": "evaluated",
        "strong_validation": False,
        "criteria": {
            "class": args.required_class,
            "min_iou": float(args.min_iou),
            "max_depth_delta_m": float(args.max_depth_delta_m),
        },
        "metrics": metrics,
        "matched_detection": md_det,
        "source_image": md_row.get("image"),
    }

    iou = float(metrics["best_iou"])
    depth_delta = metrics["depth_delta_m"]
    if md_det is not None and iou >= args.min_iou and depth_delta is not None and depth_delta <= args.max_depth_delta_m:
        validation["status"] = "strong_agree"
        validation["strong_validation"] = True
        return "A", validation, True

    if (
        args.mark_depth_conflicts_c
        and md_det is not None
        and iou >= args.min_iou
        and depth_delta is not None
        and depth_delta > args.max_depth_delta_m
    ):
        validation["status"] = "depth_conflict"
        return "C", validation, False

    validation["status"] = "insufficient_monodetr_agreement"
    return previous_grade, validation, bool(det.get("use_for_training", True))


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grade_counts = {"A": 0, "B": 0, "C": 0}
    sampled = 0
    strong = 0
    conflicts = 0
    for row in rows:
        grade_counts[str(row.get("final_label_grade") or "B")] = grade_counts.get(str(row.get("final_label_grade") or "B"), 0) + 1
        val = row.get("monodetr_validation") or {}
        if val.get("status") != "not_sampled_by_monodetr":
            sampled += 1
        if val.get("strong_validation"):
            strong += 1
        if val.get("status") == "depth_conflict":
            conflicts += 1
    return {
        "total": len(rows),
        "grade_counts": grade_counts,
        "monodetr_sampled": sampled,
        "monodetr_strong_agree": strong,
        "monodetr_depth_conflicts": conflicts,
        "use_for_training": sum(bool(r.get("use_for_training", True)) for r in rows),
    }


def _process_target(target: int, indexed_md: dict[tuple[int, str], dict[str, Any]], args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    baseline_path = args.baseline_dir / f"4_15_target{target}_final_labels_baseline.json"
    obj = _load_json(baseline_path)
    upgraded = deepcopy(obj)
    flat_rows: list[dict[str, Any]] = []

    for frame in upgraded.get("frames", []):
        image_name = _image_name(frame.get("image"))
        md_row = indexed_md.get((target, image_name))
        final_dets = []
        for det in frame.get("final_detections") or []:
            new_grade, validation, use_for_training = _validate_detection(det, md_row, args)
            d = dict(det)
            old_grade = str(d.get("final_label_grade") or "B")
            d["monodetr_validation"] = validation
            d["monodetr_grade_before"] = old_grade
            d["final_label_grade"] = new_grade
            d["final_label_reason"] = (
                "monodetr_strong_validation"
                if new_grade == "A" and validation.get("strong_validation")
                else "monodetr_depth_conflict"
                if new_grade == "C" and validation.get("status") == "depth_conflict"
                else d.get("final_label_reason", "kept_existing_grade")
            )
            d["use_for_training"] = bool(use_for_training)
            # Deliberately keep final_vehicle_center_m unchanged.
            final_dets.append(d)
            flat_rows.append(d)
        frame["final_detections"] = final_dets
        if final_dets:
            best = max(final_dets, key=lambda x: (_grade_rank(x.get("final_label_grade")), float(x.get("label_confidence") or 0.0)))
            frame["final_label_grade"] = best.get("final_label_grade")
            frame["final_vehicle_center_m"] = best.get("final_vehicle_center_m")
            frame["final_status"] = "ok"

    upgraded.setdefault("meta", {})
    upgraded["meta"]["monodetr_grade_upgrade"] = {
        "monodetr_json": str(args.monodetr_json),
        "min_iou": float(args.min_iou),
        "max_depth_delta_m": float(args.max_depth_delta_m),
        "required_class": args.required_class,
        "center_policy": "final_vehicle_center_m is preserved; MonoDETR is validation only",
        "mark_depth_conflicts_c": bool(args.mark_depth_conflicts_c),
    }
    summary = {"target": target, "baseline_json": str(baseline_path), **_summarize_rows(flat_rows)}
    return upgraded, summary


def main() -> None:
    args = parse_args()
    monodetr = _load_json(args.monodetr_json)
    indexed_md = _index_monodetr(monodetr)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for target in args.targets:
        upgraded, summary = _process_target(target, indexed_md, args)
        output_path = args.output_dir / f"4_15_target{target}_final_labels_monodetr_upgraded.json"
        _write_json(output_path, upgraded)
        summary["output_json"] = str(output_path)
        summaries.append(summary)
        print(
            f"Target{target}: total={summary['total']} "
            f"A={summary['grade_counts'].get('A', 0)} B={summary['grade_counts'].get('B', 0)} "
            f"C={summary['grade_counts'].get('C', 0)} strong={summary['monodetr_strong_agree']}"
        )

    total_counts = {"A": 0, "B": 0, "C": 0}
    for s in summaries:
        for grade, count in s["grade_counts"].items():
            total_counts[grade] = total_counts.get(grade, 0) + int(count)
    combined = {
        "monodetr_json": str(args.monodetr_json),
        "output_dir": str(args.output_dir),
        "criteria": {
            "class": args.required_class,
            "min_iou": float(args.min_iou),
            "max_depth_delta_m": float(args.max_depth_delta_m),
            "mark_depth_conflicts_c": bool(args.mark_depth_conflicts_c),
        },
        "total_grade_counts": total_counts,
        "total_labels": int(sum(s["total"] for s in summaries)),
        "monodetr_strong_agree": int(sum(s["monodetr_strong_agree"] for s in summaries)),
        "monodetr_depth_conflicts": int(sum(s["monodetr_depth_conflicts"] for s in summaries)),
        "targets": summaries,
    }
    _write_json(args.summary_json, combined)
    print(f"Wrote {args.summary_json}")


if __name__ == "__main__":
    main()
