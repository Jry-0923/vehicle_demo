import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Align MoGe/MoGe-lite outputs into the project pseudo-label coordinate frame.")
    p.add_argument("--moge-completion-json", type=Path, required=True)
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--bev-png", type=Path, required=True)
    p.add_argument("--mode", choices=["uniform", "diagonal", "affine", "zonly"], default="uniform")
    p.add_argument("--fit-source", choices=["visible", "completed"], default="visible")
    p.add_argument("--robust-iters", type=int, default=3)
    p.add_argument("--trim-quantile", type=float, default=0.90)
    p.add_argument("--min-fit-pairs", type=int, default=12)
    p.add_argument("--bev-width", type=int, default=1300)
    p.add_argument("--bev-height", type=int, default=900)
    p.add_argument("--bev-margin", type=int, default=70)
    return p.parse_args()


def _point(values) -> Optional[np.ndarray]:
    if not isinstance(values, list) or len(values) != 3:
        return None
    arr = np.asarray(values, dtype=np.float64)
    if not np.isfinite(arr).all():
        return None
    return arr


def _collect_pairs(obj: Dict, fit_source: str) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    x_rows: List[np.ndarray] = []
    y_rows: List[np.ndarray] = []
    indices: List[int] = []
    key = "visible_surface_center_m" if fit_source == "visible" else "completed_box_center_m"
    for idx, det in enumerate(obj.get("detections", [])):
        comp = det.get("completion") or {}
        if comp.get("status") != "ok":
            continue
        x = _point(comp.get(key))
        y = _point(det.get("source_center_m"))
        if x is None or y is None:
            continue
        x_rows.append(x)
        y_rows.append(y)
        indices.append(idx)
    if not x_rows:
        return np.zeros((0, 3)), np.zeros((0, 3)), []
    return np.vstack(x_rows), np.vstack(y_rows), indices


def _fit_transform(X: np.ndarray, Y: np.ndarray, mode: str) -> Dict:
    if mode == "uniform":
        x_mu = X.mean(axis=0)
        y_mu = Y.mean(axis=0)
        Xc = X - x_mu
        Yc = Y - y_mu
        scale = float(np.sum(Xc * Yc) / max(float(np.sum(Xc * Xc)), 1e-12))
        translation = y_mu - scale * x_mu
        return {"mode": mode, "scale": scale, "translation": translation.tolist()}

    if mode == "diagonal":
        rows = []
        for i in range(3):
            a, b = np.linalg.lstsq(np.c_[X[:, i], np.ones(len(X))], Y[:, i], rcond=None)[0]
            rows.append([float(a), float(b)])
        return {"mode": mode, "axis_scale_offset": rows}

    if mode == "zonly":
        a, b = np.linalg.lstsq(np.c_[X[:, 2], np.ones(len(X))], Y[:, 2], rcond=None)[0]
        return {"mode": mode, "z_scale": float(a), "z_offset": float(b)}

    A = np.c_[X, np.ones(len(X))]
    matrix = np.linalg.lstsq(A, Y, rcond=None)[0]
    return {"mode": mode, "affine_4x3": matrix.tolist()}


def _apply_transform_point(pt: np.ndarray, transform: Dict) -> np.ndarray:
    mode = transform["mode"]
    if mode == "uniform":
        return pt * float(transform["scale"]) + np.asarray(transform["translation"], dtype=np.float64)
    if mode == "diagonal":
        rows = np.asarray(transform["axis_scale_offset"], dtype=np.float64)
        return pt * rows[:, 0] + rows[:, 1]
    if mode == "zonly":
        out = pt.copy()
        out[2] = out[2] * float(transform["z_scale"]) + float(transform["z_offset"])
        return out
    matrix = np.asarray(transform["affine_4x3"], dtype=np.float64)
    return np.r_[pt, 1.0] @ matrix


def _predict(X: np.ndarray, transform: Dict) -> np.ndarray:
    return np.vstack([_apply_transform_point(x, transform) for x in X])


def _fit_robust(X: np.ndarray, Y: np.ndarray, mode: str, iters: int, trim_q: float, min_pairs: int) -> Tuple[Dict, np.ndarray]:
    keep = np.ones(len(X), dtype=bool)
    transform = _fit_transform(X, Y, mode)
    trim_q = max(0.5, min(1.0, float(trim_q)))
    for _ in range(max(0, int(iters))):
        transform = _fit_transform(X[keep], Y[keep], mode)
        err = np.linalg.norm(_predict(X, transform) - Y, axis=1)
        threshold = float(np.quantile(err[keep], trim_q))
        next_keep = err <= threshold
        if int(next_keep.sum()) < int(min_pairs):
            break
        keep = next_keep
    transform = _fit_transform(X[keep], Y[keep], mode)
    return transform, keep


def _err_stats(pred: np.ndarray, target: np.ndarray) -> Dict:
    err = np.linalg.norm(pred - target, axis=1)
    return {
        "count": int(len(err)),
        "median_m": float(np.median(err)) if len(err) else None,
        "mean_m": float(np.mean(err)) if len(err) else None,
        "p90_m": float(np.quantile(err, 0.90)) if len(err) else None,
        "max_m": float(np.max(err)) if len(err) else None,
    }


def _align_detection(det: Dict, transform: Dict) -> Dict:
    out = json.loads(json.dumps(det))
    comp = out.get("completion") or {}
    if comp.get("status") != "ok":
        return out

    for key in ("visible_surface_center_m", "completed_box_center_m"):
        pt = _point(comp.get(key))
        if pt is not None:
            comp[f"aligned_{key}"] = [float(v) for v in _apply_transform_point(pt, transform).tolist()]

    visible = _point(comp.get("visible_surface_center_m"))
    corners = comp.get("completed_box_bev_corners_xz")
    if visible is not None and isinstance(corners, list):
        aligned = []
        for corner in corners:
            if not isinstance(corner, list) or len(corner) != 2:
                continue
            lifted = np.asarray([corner[0], visible[1], corner[1]], dtype=np.float64)
            mapped = _apply_transform_point(lifted, transform)
            aligned.append([float(mapped[0]), float(mapped[2])])
        comp["aligned_completed_box_bev_corners_xz"] = aligned

    src = _point(out.get("source_center_m"))
    aligned_completed = _point(comp.get("aligned_completed_box_center_m"))
    aligned_visible = _point(comp.get("aligned_visible_surface_center_m"))
    if src is not None:
        if aligned_visible is not None:
            comp["aligned_visible_to_source_error_m"] = float(np.linalg.norm(aligned_visible - src))
        if aligned_completed is not None:
            comp["aligned_completed_to_source_error_m"] = float(np.linalg.norm(aligned_completed - src))
    out["completion"] = comp
    return out


def _draw_bev(rows: List[Dict], out_png: Path, width: int, height: int, margin: int) -> None:
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    points = []
    for det in rows:
        comp = det.get("completion") or {}
        src = _point(det.get("source_center_m"))
        av = _point(comp.get("aligned_visible_surface_center_m"))
        ac = _point(comp.get("aligned_completed_box_center_m"))
        if src is not None:
            points.append(src[[0, 2]])
        if av is not None:
            points.append(av[[0, 2]])
        if ac is not None:
            points.append(ac[[0, 2]])
        for corner in comp.get("aligned_completed_box_bev_corners_xz") or []:
            if isinstance(corner, list) and len(corner) == 2:
                points.append(np.asarray(corner, dtype=np.float64))
    if not points:
        cv2.putText(img, "No aligned points", (margin, margin), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_png), img)
        return

    arr = np.vstack(points)
    x_min, z_min = np.min(arr, axis=0)
    x_max, z_max = np.max(arr, axis=0)
    pad_x = max(1.0, float(x_max - x_min) * 0.08)
    pad_z = max(1.0, float(z_max - z_min) * 0.08)
    x_min -= pad_x
    x_max += pad_x
    z_min -= pad_z
    z_max += pad_z

    def to_px(x: float, z: float) -> Tuple[int, int]:
        px = margin + (x - x_min) / max(x_max - x_min, 1e-6) * max(width - 2 * margin, 1)
        py = height - margin - (z - z_min) / max(z_max - z_min, 1e-6) * max(height - 2 * margin, 1)
        return int(round(px)), int(round(py))

    for gx in np.linspace(x_min, x_max, 7):
        x0, _ = to_px(float(gx), z_min)
        cv2.line(img, (x0, margin), (x0, height - margin), (235, 235, 235), 1)
    for gz in np.linspace(z_min, z_max, 7):
        _, y0 = to_px(x_min, float(gz))
        cv2.line(img, (margin, y0), (width - margin, y0), (235, 235, 235), 1)

    for det in rows:
        comp = det.get("completion") or {}
        corners = comp.get("aligned_completed_box_bev_corners_xz") or []
        if len(corners) >= 4:
            px = [to_px(float(x), float(z)) for x, z in corners[:4]]
            for a, b in zip(px, px[1:] + px[:1]):
                cv2.line(img, a, b, (0, 160, 0), 1)
        src = _point(det.get("source_center_m"))
        av = _point(comp.get("aligned_visible_surface_center_m"))
        ac = _point(comp.get("aligned_completed_box_center_m"))
        if src is not None:
            cv2.circle(img, to_px(src[0], src[2]), 4, (0, 0, 220), -1)
        if av is not None:
            cv2.circle(img, to_px(av[0], av[2]), 4, (200, 110, 0), -1)
        if ac is not None:
            p = to_px(ac[0], ac[2])
            cv2.circle(img, p, 4, (0, 150, 0), -1)
            cv2.putText(img, str(det.get("frame_index", "")), (p[0] + 5, p[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 120, 0), 1)

    cv2.putText(
        img,
        "Aligned BEV: red=source pseudo label, blue=aligned visible MoGe, green=aligned completed box",
        (margin, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (0, 0, 0),
        2,
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), img)


def main() -> None:
    args = parse_args()
    with args.moge_completion_json.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    X, Y, pair_indices = _collect_pairs(obj, args.fit_source)
    if len(X) < args.min_fit_pairs:
        raise RuntimeError(f"Only {len(X)} fit pairs found; need at least {args.min_fit_pairs}")

    transform, keep = _fit_robust(X, Y, args.mode, args.robust_iters, args.trim_quantile, args.min_fit_pairs)
    pred_raw = X
    pred_aligned = _predict(X, transform)

    aligned = [_align_detection(det, transform) for det in obj.get("detections", [])]
    comp_pairs = []
    vis_pairs = []
    for det in aligned:
        comp = det.get("completion") or {}
        src = _point(det.get("source_center_m"))
        av = _point(comp.get("aligned_visible_surface_center_m"))
        ac = _point(comp.get("aligned_completed_box_center_m"))
        if src is not None and av is not None:
            vis_pairs.append((av, src))
        if src is not None and ac is not None:
            comp_pairs.append((ac, src))

    payload = {
        "meta": {
            **(obj.get("meta") or {}),
            "coordinate_alignment": {
                "reference": "source_center_m from current image-dominant temporal pseudo labels",
                "fit_source": args.fit_source,
                "mode": args.mode,
                "robust_iters": int(args.robust_iters),
                "trim_quantile": float(args.trim_quantile),
                "fit_pairs_total": int(len(X)),
                "fit_pairs_used": int(keep.sum()),
                "transform": transform,
                "fit_source_raw_error": _err_stats(pred_raw, Y),
                "fit_source_aligned_error": _err_stats(pred_aligned, Y),
                "aligned_visible_error": _err_stats(np.vstack([a for a, _ in vis_pairs]), np.vstack([b for _, b in vis_pairs])) if vis_pairs else None,
                "aligned_completed_error": _err_stats(np.vstack([a for a, _ in comp_pairs]), np.vstack([b for _, b in comp_pairs])) if comp_pairs else None,
            },
        },
        "detections": aligned,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    _draw_bev(aligned, args.bev_png, args.bev_width, args.bev_height, args.bev_margin)

    a = payload["meta"]["coordinate_alignment"]
    print(
        f"Aligned MoGe using {args.mode}/{args.fit_source}: "
        f"raw median={a['fit_source_raw_error']['median_m']:.3f}m, "
        f"aligned median={a['fit_source_aligned_error']['median_m']:.3f}m -> "
        f"{args.output_json}, {args.bev_png}"
    )


if __name__ == "__main__":
    main()
