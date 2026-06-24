import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare temporal image pseudo-GT centers with Mono3D outputs and assign A/B/C label grades.")
    p.add_argument("--temporal-json", type=Path, required=True, help="Output from filter_image_gt_tracks.py")
    p.add_argument("--mono3d-json", type=Path, default=None, help="Mono3D detections JSON. Optional: if omitted, temporal labels become B grade.")
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--summary-json", type=Path, required=True)
    p.add_argument("--min-iou", type=float, default=0.30)
    p.add_argument("--a-center-dist-m", type=float, default=1.0)
    p.add_argument("--b-center-dist-m", type=float, default=2.0)
    p.add_argument("--min-mono3d-score", type=float, default=0.20)
    return p.parse_args()


def _point(v) -> Optional[np.ndarray]:
    if not isinstance(v, list) or len(v) != 3:
        return None
    pt = np.asarray(v, dtype=np.float32)
    if not np.isfinite(pt).all():
        return None
    return pt


def _box(v) -> Optional[List[float]]:
    if not isinstance(v, list) or len(v) != 4:
        return None
    try:
        b = [float(x) for x in v]
    except Exception:
        return None
    if not all(np.isfinite(b)):
        return None
    if b[2] <= b[0] or b[3] <= b[1]:
        return None
    return b


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return 0.0 if denom <= 0 else float(inter / denom)


def _score(det: Dict) -> Optional[float]:
    for key in ("mono3d_score", "score", "conf", "confidence", "det_conf"):
        if det.get(key) is not None:
            return float(det[key])
    return None


def _mono_center(det: Dict) -> Optional[np.ndarray]:
    for key in ("mono3d_center_m", "center_m", "bbox3d_center_m", "location", "center_3d_m"):
        pt = _point(det.get(key))
        if pt is not None:
            return pt
    box3d = det.get("box_3d") or det.get("bbox_3d")
    if isinstance(box3d, dict):
        for key in ("center_m", "center", "location"):
            pt = _point(box3d.get(key))
            if pt is not None:
                return pt
    return None


def _load_mono3d(path: Optional[Path]) -> Dict[int, List[Dict]]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    by_frame: Dict[int, List[Dict]] = {}
    for i, fr in enumerate(obj.get("frames", [])):
        frame_index = int(fr.get("frame_index", i + 1))
        dets = []
        for det in fr.get("detections") or fr.get("mono3d_detections") or []:
            center = _mono_center(det)
            box = _box(det.get("box_xyxy") or det.get("box_2d") or det.get("bbox"))
            if center is None or box is None:
                continue
            score = _score(det)
            dets.append({"raw": det, "center": center, "box": box, "score": score})
        by_frame[frame_index] = dets
    return by_frame


def _best_mono_match(det: Dict, mono_dets: List[Dict], min_iou: float, min_score: float) -> Tuple[Optional[Dict], float]:
    box = _box(det.get("box_xyxy"))
    if box is None:
        return None, 0.0
    best = None
    best_iou = 0.0
    for md in mono_dets:
        score = md.get("score")
        if score is not None and score < min_score:
            continue
        iou = _iou(box, md["box"])
        if iou > best_iou:
            best = md
            best_iou = iou
    if best is None or best_iou < min_iou:
        return None, best_iou
    return best, best_iou


def _grade(dist: Optional[float], has_mono: bool, a_dist: float, b_dist: float) -> Tuple[str, str, bool]:
    if not has_mono:
        return "B", "temporal_only_no_mono3d_match", True
    if dist is None:
        return "B", "temporal_only_mono3d_missing_center", True
    if dist <= a_dist:
        return "A", "mono3d_temporal_agree", True
    if dist <= b_dist:
        return "B", "mono3d_temporal_near", True
    return "C", "mono3d_temporal_disagree", False


def main() -> None:
    args = parse_args()
    with args.temporal_json.open("r", encoding="utf-8") as f:
        temporal = json.load(f)
    mono_by_frame = _load_mono3d(args.mono3d_json)

    frames = json.loads(json.dumps(temporal.get("frames", [])))
    counts = {"A": 0, "B": 0, "C": 0}
    total = 0
    matched = 0
    dists: List[float] = []

    for fr in frames:
        frame_index = int(fr.get("frame_index", 0))
        mono_dets = mono_by_frame.get(frame_index, [])
        final_dets = []
        for det in fr.get("filtered_detections") or []:
            center = _point(det.get("smoothed_pseudo_gt_vehicle_center_m") or det.get("pseudo_gt_vehicle_center_m"))
            if center is None:
                continue
            match, match_iou = _best_mono_match(det, mono_dets, args.min_iou, args.min_mono3d_score)
            dist = None
            mono_info = None
            if match is not None:
                matched += 1
                dist = float(np.linalg.norm(match["center"] - center))
                dists.append(dist)
                mono_info = {
                    "box_xyxy": [float(v) for v in match["box"]],
                    "center_m": [float(v) for v in match["center"].tolist()],
                    "score": match.get("score"),
                    "iou_2d": float(match_iou),
                    "center_distance_m": dist,
                    "raw": match["raw"],
                }
            grade, reason, use_for_training = _grade(dist, match is not None, args.a_center_dist_m, args.b_center_dist_m)
            d = dict(det)
            d["mono3d_validation"] = mono_info
            d["final_label_grade"] = grade
            d["final_label_reason"] = reason
            d["use_for_training"] = bool(use_for_training)
            d["final_vehicle_center_m"] = [float(v) for v in center.tolist()]
            final_dets.append(d)
            counts[grade] += 1
            total += 1
        fr["final_detections"] = final_dets
        if final_dets:
            best = max(final_dets, key=lambda x: (x.get("final_label_grade") == "A", float(x.get("label_confidence") or 0.0)))
            fr["final_status"] = "ok"
            fr["final_vehicle_center_m"] = best.get("final_vehicle_center_m")
            fr["final_label_grade"] = best.get("final_label_grade")
        else:
            fr["final_status"] = "no_final_label"

    payload = {
        "meta": {
            **(temporal.get("meta") or {}),
            "mono3d_comparison": {
                "mono3d_json": None if args.mono3d_json is None else str(args.mono3d_json),
                "min_iou": float(args.min_iou),
                "a_center_dist_m": float(args.a_center_dist_m),
                "b_center_dist_m": float(args.b_center_dist_m),
                "min_mono3d_score": float(args.min_mono3d_score),
                "grade_definitions": {
                    "A": "temporal image pseudo-GT and Mono3D center agree",
                    "B": "temporal image pseudo-GT is stable but Mono3D is missing or only near",
                    "C": "Mono3D and temporal image pseudo-GT disagree; do not use for training",
                },
            },
        },
        "frames": frames,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    summary = {
        "temporal_json": str(args.temporal_json),
        "mono3d_json": None if args.mono3d_json is None else str(args.mono3d_json),
        "total_final_detections": int(total),
        "mono3d_matched_detections": int(matched),
        "grade_counts": counts,
        "use_for_training_count": int(counts["A"] + counts["B"]),
        "mono3d_center_distance_m": {
            "mean": float(np.mean(dists)) if dists else None,
            "median": float(np.median(dists)) if dists else None,
            "max": float(np.max(dists)) if dists else None,
        },
        "output_json": str(args.output_json),
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(
        f"Compared temporal labels with Mono3D: total={total} matched={matched} "
        f"A={counts['A']} B={counts['B']} C={counts['C']} -> {args.output_json}"
    )


if __name__ == "__main__":
    main()
