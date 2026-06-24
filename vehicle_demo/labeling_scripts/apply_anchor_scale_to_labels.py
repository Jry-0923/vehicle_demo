#!/usr/bin/env python3
"""Apply a global anchor-derived metric scale to 4.15 label JSON files."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any


CENTER_KEYS = [
    "pseudo_gt_vehicle_center_m",
    "smoothed_pseudo_gt_vehicle_center_m",
    "temporal_pseudo_gt_vehicle_center_m",
    "final_vehicle_center_m",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Apply anchor scale correction to label JSON center fields.")
    p.add_argument("--scale-json", type=Path, required=True)
    p.add_argument(
        "--input-dir",
        type=Path,
        default=Path("outputs/monodetr_upgraded_50m_all_strong_only"),
        help="Directory containing 4_15_targetN_final_labels_monodetr_upgraded.json.",
    )
    p.add_argument("--output-dir", type=Path, default=Path("outputs/anchor_scaled_labels"))
    p.add_argument("--targets", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7])
    p.add_argument("--scale-key", type=str, default="scale_median")
    return p.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _scale_point(v: Any, scale: float) -> list[float] | None:
    if not isinstance(v, list) or len(v) < 3:
        return None
    try:
        out = [float(v[0]) * scale, float(v[1]) * scale, float(v[2]) * scale]
    except Exception:
        return None
    return out


def _apply_to_record(record: dict[str, Any], scale: float) -> int:
    changed = 0
    for key in CENTER_KEYS:
        scaled = _scale_point(record.get(key), scale)
        if scaled is None:
            continue
        record[f"scale_corrected_{key}"] = scaled
        changed += 1
    return changed


def main() -> None:
    args = parse_args()
    scale_obj = _load_json(args.scale_json)
    if args.scale_key not in scale_obj:
        raise KeyError(f"{args.scale_key!r} not found in {args.scale_json}")
    scale = float(scale_obj[args.scale_key])
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for target in args.targets:
        in_path = args.input_dir / f"4_15_target{target}_final_labels_monodetr_upgraded.json"
        if not in_path.exists():
            in_path = args.input_dir / f"4_15_target{target}_final_labels_baseline.json"
        if not in_path.exists():
            summaries.append({"target": target, "status": "missing", "input_json": str(in_path)})
            continue

        obj = deepcopy(_load_json(in_path))
        changed = 0
        for frame in obj.get("frames", []):
            changed += _apply_to_record(frame, scale)
            for key in ("detections", "filtered_detections", "final_detections"):
                for det in frame.get(key) or []:
                    changed += _apply_to_record(det, scale)

        obj["meta"] = {
            **(obj.get("meta") or {}),
            "anchor_scale_correction": {
                "scale": scale,
                "scale_key": args.scale_key,
                "scale_json": str(args.scale_json),
                "policy": "Uniform metric scale applied to camera-frame 3D center fields. Image projections are unchanged.",
            },
        }
        out_path = args.output_dir / f"4_15_target{target}_final_labels_anchor_scaled.json"
        _write_json(out_path, obj)
        summaries.append(
            {
                "target": target,
                "status": "ok",
                "input_json": str(in_path),
                "output_json": str(out_path),
                "scale": scale,
                "scaled_fields_written": int(changed),
            }
        )

    summary = {
        "task": "apply_anchor_scale_to_labels",
        "scale_json": str(args.scale_json),
        "scale_key": args.scale_key,
        "scale": scale,
        "targets": summaries,
    }
    _write_json(args.output_dir / "anchor_scaled_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
