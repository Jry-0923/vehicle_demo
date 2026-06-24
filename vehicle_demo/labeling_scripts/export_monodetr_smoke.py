#!/usr/bin/env python3
"""Run MonoDETR on sampled project frames and export inverse-affine 3D box predictions."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml
from PIL import Image


CLASS_NAMES = ("Pedestrian", "Car", "Cyclist")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MonoDETR smoke exporter with inverse-affine 2D box recovery.")
    p.add_argument("--monodetr-repo", type=Path, default=Path("third_party/MonoDETR"))
    p.add_argument("--checkpoint", type=Path, default=Path("models/monodetr_checkpoint.pth"))
    p.add_argument("--config", type=Path, default=Path("third_party/MonoDETR/configs/monodetr.yaml"))
    p.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    p.add_argument("--targets", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7])
    p.add_argument("--samples-per-target", type=int, default=15)
    p.add_argument("--threshold", type=float, default=0.05)
    p.add_argument("--resolution", type=int, nargs=2, default=[1280, 384], metavar=("W", "H"))
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--overlay-jpg", type=Path, default=None)
    p.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    return p.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _iou_xyxy(a: list[float] | None, b: list[float] | None) -> float | None:
    if a is None or b is None:
        return None
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


def _pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    return torch.device(name)


def _select_samples(outputs_dir: Path, targets: list[int], samples_per_target: int) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for target in targets:
        path = outputs_dir / f"4_15_target{target}_final_labels_baseline.json"
        data = _load_json(path)
        candidates = []
        for frame in data.get("frames", []):
            detections = frame.get("final_detections") or []
            if not detections:
                continue
            det = detections[0]
            center = det.get("final_vehicle_center_m")
            if not (isinstance(center, list) and len(center) >= 3 and float(center[2]) <= 50.0):
                continue
            candidates.append({"target": target, "frame": frame, "gt": det})
        if not candidates:
            continue
        if len(candidates) <= samples_per_target:
            selected = candidates
        else:
            idxs = np.linspace(0, len(candidates) - 1, samples_per_target).round().astype(int)
            selected = [candidates[int(i)] for i in idxs]
        samples.extend(selected)
    return samples


def _build_model(args: argparse.Namespace, device: torch.device):
    repo = args.monodetr_repo.resolve()
    sys.path.insert(0, str(repo))
    from lib.helpers.model_helper import build_model

    cfg = yaml.load(args.config.open("r", encoding="utf-8"), Loader=yaml.Loader)
    cfg["model"]["device"] = str(device)
    model, _ = build_model(cfg["model"])
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model


def _calibration(outputs_dir: Path, target: int, img_size: np.ndarray, resolution: np.ndarray, scale_p2: bool = False):
    sys.path.insert(0, str(Path("third_party/MonoDETR").resolve()))
    from lib.datasets.kitti.kitti_utils import Calibration

    obj = _load_json(outputs_dir / f"calib_4.15_target{target}_auto.json")
    k = np.asarray(obj["K"], dtype=np.float32)
    p2 = np.zeros((3, 4), dtype=np.float32)
    p2[:3, :3] = k
    if scale_p2:
        sx, sy = float(resolution[0]) / float(img_size[0]), float(resolution[1]) / float(img_size[1])
        p2 = np.asarray([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=np.float32) @ p2
    calib = Calibration(
        {
            "P2": p2,
            "R0": np.eye(3, dtype=np.float32),
            "Tr_velo2cam": np.concatenate([np.eye(3, dtype=np.float32), np.zeros((3, 1), dtype=np.float32)], axis=1),
        }
    )
    return p2, calib


def _preprocess(image_path: Path, resolution: np.ndarray):
    from lib.datasets.kitti.kitti_utils import get_affine_transform

    image = Image.open(image_path).convert("RGB")
    img_size = np.asarray(image.size, dtype=np.float32)
    _, trans_inv = get_affine_transform(img_size / 2, img_size, 0, resolution, inv=1)
    image = image.transform(
        tuple(resolution.tolist()),
        method=Image.AFFINE,
        data=tuple(trans_inv.reshape(-1).tolist()),
        resample=Image.BILINEAR,
    )
    arr = np.asarray(image).astype(np.float32) / 255.0
    arr = (arr - np.asarray([0.485, 0.456, 0.406], dtype=np.float32)) / np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).float()
    return tensor, img_size, trans_inv


def _decode(outputs, calib, img_size: np.ndarray, resolution: np.ndarray, trans_inv: np.ndarray, threshold: float):
    from lib.datasets.kitti.kitti_utils import affine_transform
    from lib.helpers.decode_helper import decode_detections, extract_dets_from_outputs

    dets = extract_dets_from_outputs(outputs=outputs, K=50, topk=50).detach().cpu().numpy()
    info = {
        "img_id": np.asarray([0]),
        "img_size": np.asarray([img_size], dtype=np.float32),
        "bbox_downsample_ratio": np.asarray([[1, 1]], dtype=np.float32),
    }
    decoded = decode_detections(dets, info, [calib], np.zeros((3, 3), dtype=np.float32), threshold=threshold).get(0, [])
    rows = []
    for raw, pred in zip(dets[0], decoded):
        cls_id = int(pred[0])
        cx, cy, w, h = [float(x) for x in raw[2:6]]
        model_box = np.asarray(
            [(cx - w / 2) * resolution[0], (cy - h / 2) * resolution[1], (cx + w / 2) * resolution[0], (cy + h / 2) * resolution[1]],
            dtype=np.float32,
        )
        p1 = affine_transform(model_box[:2], trans_inv)
        p2 = affine_transform(model_box[2:], trans_inv)
        hwl = [float(x) for x in pred[6:9]]
        loc = [float(x) for x in pred[9:12]]
        rows.append(
            {
                "class": CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id),
                "score": float(pred[13]),
                "bbox_2d": [float(p1[0]), float(p1[1]), float(p2[0]), float(p2[1])],
                "bbox_2d_raw_decode": [float(x) for x in pred[2:6]],
                "center_3d": loc,
                "size_3d": [hwl[2], hwl[1], hwl[0]],
                "size_3d_hwl": hwl,
                "yaw_rotation_y": float(pred[12]),
                "alpha": float(pred[1]),
                "depth": float(loc[2]),
                "camera_coordinate_convention": "KITTI rect camera: x right, y down, z forward",
            }
        )
    return rows


def _best_match(detections: list[dict[str, Any]], gt_box: list[float]) -> tuple[dict[str, Any] | None, float]:
    best = None
    best_iou = 0.0
    for det in detections:
        val = _iou_xyxy(det.get("bbox_2d"), gt_box) or 0.0
        if val > best_iou:
            best = det
            best_iou = val
    return best, best_iou


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    detected = [r for r in rows if r.get("best_detection")]
    ious = [float(r["best_iou"]) for r in rows]
    deltas = [float(r["depth_delta_m"]) for r in rows if r.get("depth_delta_m") is not None]
    return {
        "frames": len(rows),
        "detected_frames": len(detected),
        "iou_ge_0_5": sum(i >= 0.5 for i in ious),
        "mean_iou": float(np.mean(ious)) if ious else None,
        "vehicle_class_matches": sum((r.get("best_detection") or {}).get("class") == "Car" for r in rows),
        "depth_delta_le_5m": sum(d <= 5.0 for d in deltas),
        "mean_depth_delta_m": float(np.mean(deltas)) if deltas else None,
    }


def _summarize_by_target(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["target"]), []).append(row)
    return {target: _summarize(target_rows) for target, target_rows in sorted(grouped.items(), key=lambda x: int(x[0]))}


def _draw_overlay(rows: list[dict[str, Any]], output_path: Path) -> None:
    thumbs = []
    for row in rows[:21]:
        image_path = Path(row["image"])
        img = cv2.imread(str(image_path))
        if img is None:
            continue
        img = cv2.resize(img, (480, 270))
        sx, sy = 480 / row["image_width"], 270 / row["image_height"]

        def draw(box, color, label):
            if not box:
                return
            x1, y1, x2, y2 = [int(round(v * sx if i % 2 == 0 else v * sy)) for i, v in enumerate(box)]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label, (max(0, x1), max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        draw(row["gt_box"], (0, 255, 0), "GT")
        best = row.get("best_detection") or {}
        draw(best.get("bbox_2d"), (255, 0, 0), "MonoDETR")
        text = f"T{row['target']} IoU={row['best_iou']:.2f}"
        if row.get("depth_delta_m") is not None:
            text += f" dZ={row['depth_delta_m']:.1f}"
        cv2.putText(img, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        thumbs.append(img)
    if not thumbs:
        return
    cols = 3
    blank = np.zeros_like(thumbs[0])
    while len(thumbs) % cols:
        thumbs.append(blank.copy())
    canvas = np.vstack([np.hstack(thumbs[i : i + cols]) for i in range(0, len(thumbs), cols)])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)


def main() -> None:
    args = parse_args()
    os.environ.setdefault("NUMBA_ENABLE_CUDASIM", "1")
    device = _pick_device(args.device)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    if args.overlay_jpg is not None:
        args.overlay_jpg.parent.mkdir(parents=True, exist_ok=True)

    samples = _select_samples(args.outputs_dir, args.targets, args.samples_per_target)
    model = _build_model(args, device)
    resolution = np.asarray(args.resolution, dtype=np.float32)

    frames = []
    rows = []
    with torch.no_grad():
        for sample in samples:
            target = int(sample["target"])
            frame = sample["frame"]
            gt = sample["gt"]
            image_path = Path(frame["image"])
            if not image_path.is_absolute():
                image_path = Path.cwd() / image_path
            image_tensor, img_size, trans_inv = _preprocess(image_path, resolution.astype(np.int64))
            p2, calib = _calibration(args.outputs_dir, target, img_size, resolution, scale_p2=False)
            outputs = model(
                image_tensor.to(device),
                torch.from_numpy(p2).unsqueeze(0).float().to(device),
                None,
                torch.from_numpy(img_size).unsqueeze(0).float().to(device),
                dn_args=0,
            )
            detections = _decode(outputs, calib, img_size, resolution, trans_inv, args.threshold)
            best, best_iou = _best_match(detections, gt["box_xyxy"])
            gt_center = gt.get("final_vehicle_center_m") or gt.get("smoothed_pseudo_gt_vehicle_center_m") or gt.get("pseudo_gt_vehicle_center_m")
            depth_delta = None
            if best is not None and isinstance(gt_center, list) and len(gt_center) >= 3:
                depth_delta = abs(float(best["depth"]) - float(gt_center[2]))
            row = {
                "target": target,
                "image": str(image_path),
                "image_name": image_path.name,
                "image_width": float(img_size[0]),
                "image_height": float(img_size[1]),
                "gt_box": gt["box_xyxy"],
                "gt_label": gt.get("label"),
                "gt_center_3d": gt_center,
                "gt_depth_m": float(gt_center[2]) if isinstance(gt_center, list) and len(gt_center) >= 3 else None,
                "detections": detections,
                "best_iou": float(best_iou),
                "best_detection": best,
                "depth_delta_m": depth_delta,
            }
            rows.append(row)
            frames.append({"image": str(image_path), "target": target, "detections": detections})
            print(f"Target{target} {image_path.name}: dets={len(detections)} best_iou={best_iou:.3f}")

    payload = {
        "task": "monodetr_inverse_affine_smoke",
        "meta": {
            "monodetr_repo": str(args.monodetr_repo),
            "checkpoint": str(args.checkpoint),
            "resolution_wh": [int(resolution[0]), int(resolution[1])],
            "threshold": float(args.threshold),
            "samples_per_target": int(args.samples_per_target),
            "box_mapping": "inverse_affine_from_model_resolution_to_original_image",
            "p2_scaling": "not_scaled",
        },
        "summary": _summarize(rows),
        "per_target": _summarize_by_target(rows),
        "evaluation": rows,
        "frames": frames,
    }
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    if args.overlay_jpg is not None:
        _draw_overlay(rows, args.overlay_jpg)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
