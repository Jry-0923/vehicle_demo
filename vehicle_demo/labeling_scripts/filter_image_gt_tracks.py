import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Filter image pseudo-GT centers with simple temporal 3D tracking.")
    p.add_argument("--centers-json", type=Path, required=True)
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--summary-json", type=Path, required=True)
    p.add_argument("--bev-png", type=Path, required=True)
    p.add_argument("--min-label-confidence", type=float, default=0.72)
    p.add_argument("--max-link-dist-m", type=float, default=1.8)
    p.add_argument("--max-missing-frames", type=int, default=2)
    p.add_argument("--min-track-length", type=int, default=5)
    p.add_argument("--max-step-m", type=float, default=2.0)
    p.add_argument("--smooth-window", type=int, default=3)
    p.add_argument("--bev-width", type=int, default=1200)
    p.add_argument("--bev-height", type=int, default=900)
    p.add_argument("--bev-margin", type=int, default=60)
    return p.parse_args()


def _point(v) -> Optional[np.ndarray]:
    if not isinstance(v, list) or len(v) != 3:
        return None
    pt = np.asarray(v, dtype=np.float32)
    if not np.isfinite(pt).all():
        return None
    return pt


def _collect_candidates(obj: Dict, min_conf: float) -> List[Dict]:
    rows: List[Dict] = []
    for fi, fr in enumerate(obj.get("frames", [])):
        frame_index = int(fr.get("frame_index", fi + 1))
        for di, det in enumerate(fr.get("detections") or []):
            if det.get("status") != "ok":
                continue
            score = float(det.get("label_confidence") or 0.0)
            if score < min_conf:
                continue
            center = _point(det.get("pseudo_gt_vehicle_center_m"))
            if center is None:
                continue
            rows.append(
                {
                    "frame_index": frame_index,
                    "frame_list_index": fi,
                    "det_index": int(di),
                    "center": center,
                    "score": score,
                    "label": det.get("label"),
                    "box_xyxy": det.get("box_xyxy"),
                }
            )
    return rows


def _build_tracks(rows: List[Dict], max_link_dist: float, max_missing: int) -> List[Dict]:
    rows_by_frame: Dict[int, List[Dict]] = {}
    for r in rows:
        rows_by_frame.setdefault(r["frame_index"], []).append(r)

    tracks: List[Dict] = []
    active: List[Dict] = []
    next_id = 1

    for frame_index in sorted(rows_by_frame):
        detections = sorted(rows_by_frame[frame_index], key=lambda x: x["score"], reverse=True)
        for tr in active:
            tr["assigned"] = False

        pairs = []
        for ti, tr in enumerate(active):
            if frame_index - tr["last_frame"] > max_missing + 1:
                continue
            for di, det in enumerate(detections):
                dist = float(np.linalg.norm(det["center"] - tr["last_center"]))
                if dist <= max_link_dist:
                    pairs.append((dist, -det["score"], ti, di))
        pairs.sort()

        used_tracks = set()
        used_dets = set()
        for dist, _, ti, di in pairs:
            if ti in used_tracks or di in used_dets:
                continue
            tr = active[ti]
            det = detections[di]
            tr["items"].append({**det, "step_m": float(dist)})
            tr["last_center"] = det["center"]
            tr["last_frame"] = frame_index
            tr["assigned"] = True
            used_tracks.add(ti)
            used_dets.add(di)

        for di, det in enumerate(detections):
            if di in used_dets:
                continue
            tr = {
                "track_id": next_id,
                "items": [{**det, "step_m": None}],
                "last_center": det["center"],
                "last_frame": frame_index,
                "assigned": True,
            }
            next_id += 1
            active.append(tr)

        still_active = []
        for tr in active:
            if frame_index - tr["last_frame"] <= max_missing:
                still_active.append(tr)
            else:
                tracks.append(tr)
        active = still_active

    tracks.extend(active)
    return tracks


def _smooth_track(items: List[Dict], window: int) -> None:
    if window <= 1 or len(items) < 2:
        for it in items:
            it["smoothed_center"] = it["center"]
        return
    half = max(1, window // 2)
    centers = [it["center"] for it in items]
    for i, it in enumerate(items):
        lo = max(0, i - half)
        hi = min(len(items), i + half + 1)
        it["smoothed_center"] = np.median(np.asarray(centers[lo:hi], dtype=np.float32), axis=0).astype(np.float32)


def _accept_tracks(tracks: List[Dict], min_len: int, max_step: float, smooth_window: int) -> List[Dict]:
    accepted = []
    for tr in tracks:
        items = sorted(tr["items"], key=lambda x: x["frame_index"])
        if len(items) < min_len:
            tr["reject_reason"] = "short_track"
            continue
        steps = [float(x["step_m"]) for x in items if x.get("step_m") is not None]
        large_steps = [x for x in steps if x > max_step]
        if large_steps:
            tr["reject_reason"] = "large_step"
            tr["max_step_m"] = float(max(large_steps))
            continue
        _smooth_track(items, smooth_window)
        tr["items"] = items
        tr["track_length"] = len(items)
        tr["max_step_m"] = float(max(steps)) if steps else 0.0
        tr["median_step_m"] = float(np.median(steps)) if steps else 0.0
        tr["mean_score"] = float(np.mean([float(x["score"]) for x in items]))
        accepted.append(tr)
    return accepted


def _write_filtered_payload(obj: Dict, accepted: List[Dict], args: argparse.Namespace) -> Dict:
    frames = json.loads(json.dumps(obj.get("frames", [])))
    kept = 0
    index = {}
    for tr in accepted:
        for it in tr["items"]:
            index[(it["frame_list_index"], it["det_index"])] = (tr, it)

    for fi, fr in enumerate(frames):
        filtered = []
        for di, det in enumerate(fr.get("detections") or []):
            key = (fi, di)
            if key not in index:
                continue
            tr, it = index[key]
            d = dict(det)
            d["track_id"] = int(tr["track_id"])
            d["temporal_validation"] = {
                "status": "accepted_track",
                "track_length": int(tr["track_length"]),
                "track_max_step_m": float(tr["max_step_m"]),
                "track_median_step_m": float(tr["median_step_m"]),
                "step_from_previous_m": it.get("step_m"),
            }
            sm = it.get("smoothed_center")
            if sm is not None:
                d["smoothed_pseudo_gt_vehicle_center_m"] = [float(v) for v in sm.tolist()]
            filtered.append(d)
            kept += 1
        fr["filtered_detections"] = filtered
        if filtered:
            primary = max(filtered, key=lambda x: float(x.get("label_confidence") or 0.0))
            fr["status_temporal"] = "ok"
            fr["selected_track_id"] = primary.get("track_id")
            fr["temporal_pseudo_gt_vehicle_center_m"] = primary.get("smoothed_pseudo_gt_vehicle_center_m") or primary.get("pseudo_gt_vehicle_center_m")
            fr["temporal_selected_conf"] = primary.get("label_confidence")
        else:
            fr["status_temporal"] = "no_temporal_track"

    payload = {
        "meta": {
            **(obj.get("meta") or {}),
            "temporal_filter": {
                "min_label_confidence": float(args.min_label_confidence),
                "max_link_dist_m": float(args.max_link_dist_m),
                "max_missing_frames": int(args.max_missing_frames),
                "min_track_length": int(args.min_track_length),
                "max_step_m": float(args.max_step_m),
                "smooth_window": int(args.smooth_window),
                "kept_detections": int(kept),
                "accepted_tracks": int(len(accepted)),
            },
        },
        "frames": frames,
    }
    return payload


def _draw_bev(accepted: List[Dict], out_png: Path, width: int, height: int, margin: int) -> None:
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    pts = []
    for tr in accepted:
        for it in tr["items"]:
            pts.append(it.get("smoothed_center", it["center"]))
    if not pts:
        cv2.putText(img, "No accepted tracks", (margin, margin), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_png), img)
        return
    arr = np.asarray(pts, dtype=np.float32)
    xs, zs = arr[:, 0], arr[:, 2]
    x_min, x_max = float(xs.min()), float(xs.max())
    z_min, z_max = float(zs.min()), float(zs.max())
    if x_max - x_min < 1e-3:
        x_min -= 1.0; x_max += 1.0
    if z_max - z_min < 1e-3:
        z_min -= 1.0; z_max += 1.0

    def to_px(pt):
        x, z = float(pt[0]), float(pt[2])
        px = margin + (x - x_min) / (x_max - x_min) * max(width - 2 * margin, 1)
        py = height - margin - (z - z_min) / (z_max - z_min) * max(height - 2 * margin, 1)
        return int(round(px)), int(round(py))

    palette = [(0, 150, 0), (220, 80, 0), (180, 0, 180), (0, 140, 220), (100, 100, 0), (0, 120, 120)]
    for tr in accepted:
        color = palette[int(tr["track_id"]) % len(palette)]
        prev = None
        for it in tr["items"]:
            pt = it.get("smoothed_center", it["center"])
            p = to_px(pt)
            if prev is not None:
                cv2.line(img, prev, p, color, 2)
            cv2.circle(img, p, 5, color, -1)
            cv2.putText(img, f't{tr["track_id"]}:{it["frame_index"]}', (p[0] + 6, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            prev = p
    cv2.putText(img, "Temporal filtered BEV: accepted tracks only", (margin, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 0), 2)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), img)


def main() -> None:
    args = parse_args()
    with args.centers_json.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    candidates = _collect_candidates(obj, args.min_label_confidence)
    tracks = _build_tracks(candidates, args.max_link_dist_m, args.max_missing_frames)
    accepted = _accept_tracks(tracks, args.min_track_length, args.max_step_m, args.smooth_window)
    payload = _write_filtered_payload(obj, accepted, args)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    track_summaries = [
        {
            "track_id": int(t["track_id"]),
            "track_length": int(t["track_length"]),
            "frame_start": int(t["items"][0]["frame_index"]),
            "frame_end": int(t["items"][-1]["frame_index"]),
            "mean_score": float(t["mean_score"]),
            "max_step_m": float(t["max_step_m"]),
            "median_step_m": float(t["median_step_m"]),
        }
        for t in accepted
    ]
    summary = {
        "input_json": str(args.centers_json),
        "output_json": str(args.output_json),
        "candidate_detections": int(len(candidates)),
        "raw_tracks": int(len(tracks)),
        "accepted_tracks": int(len(accepted)),
        "kept_detections": int(sum(len(t["items"]) for t in accepted)),
        "track_summaries": track_summaries,
        "bev_png": str(args.bev_png),
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    _draw_bev(accepted, args.bev_png, args.bev_width, args.bev_height, args.bev_margin)
    print(
        f"Temporal filter: candidates={len(candidates)} raw_tracks={len(tracks)} "
        f"accepted_tracks={len(accepted)} kept={summary['kept_detections']} -> {args.output_json}"
    )


if __name__ == "__main__":
    main()
