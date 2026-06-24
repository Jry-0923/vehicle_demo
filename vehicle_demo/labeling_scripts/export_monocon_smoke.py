#!/usr/bin/env python3
"""Run a 5-image MonoCon smoke test and export normalized JSON predictions."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


CLASSES = ("pedestrian", "cyclist", "car")


def _load_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _as_float_list(values: Any) -> list[float]:
    arr = np.asarray(values, dtype=float).reshape(-1)
    return [float(x) for x in arr.tolist()]


def _iou_xyxy(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


def _dist3(a: Any, b: Any) -> float | None:
    if a is None or b is None:
        return None
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    if aa.shape != (3,) or bb.shape != (3,):
        return None
    return float(np.linalg.norm(aa - bb))


def _depth_delta(a: Any, b: Any) -> float | None:
    if a is None or b is None:
        return None
    aa = np.asarray(a, dtype=float).reshape(-1)
    bb = np.asarray(b, dtype=float).reshape(-1)
    if aa.size < 3 or bb.size < 3:
        return None
    return float(abs(aa[2] - bb[2]))


def _index_temporal(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for frame in data.get("frames", []):
        out[Path(frame.get("image", "")).name] = frame
    return out


def _index_moge(data: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for det in data.get("detections", []):
        out.setdefault(Path(det.get("image", "")).name, []).append(det)
    return out


def _best_match(box: list[float], candidates: list[dict[str, Any]], key: str) -> tuple[dict[str, Any] | None, float]:
    best = None
    best_iou = 0.0
    for cand in candidates:
        cand_box = cand.get(key)
        if cand_box is None:
            continue
        score = _iou_xyxy(box, [float(x) for x in cand_box])
        if score > best_iou:
            best = cand
            best_iou = score
    return best, best_iou


def _make_kitti_raw_calib(calib_json: Path, output_dir: Path, scale_xy: tuple[float, float] = (1.0, 1.0)) -> Path:
    calib = _load_json(calib_json)
    k = np.asarray(calib["K"], dtype=float)
    sx, sy = scale_xy
    k = k.copy()
    k[0, 0] *= sx
    k[0, 2] *= sx
    k[1, 1] *= sy
    k[1, 2] *= sy
    p2 = np.zeros((3, 4), dtype=float)
    p2[:, :3] = k
    path = output_dir / "monocon_target7_raw_calib.txt"
    with path.open("w", encoding="utf-8") as f:
        f.write("P_rect_02: " + " ".join(f"{v:.12g}" for v in p2.reshape(-1)) + "\n")
    return path


def _load_monocon(repo: Path, checkpoint: Path, device: str, threshold: float):
    sys.path.insert(0, str(repo))
    from model.detector import MonoConDetector

    test_config = {
        "topk": 30,
        "local_maximum_kernel": 3,
        "max_per_img": 30,
        "test_thres": threshold,
    }
    model = MonoConDetector(pretrained_backbone=False, test_config=test_config)
    try:
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint, map_location=device)
    state_dict = ckpt.get("state_dict", {}).get("model", ckpt.get("model", ckpt))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def _build_comparison(
    bbox: list[float],
    center: list[float],
    frame_name: str,
    temporal_by_image: dict[str, dict[str, Any]],
    moge_by_image: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    temporal = temporal_by_image.get(frame_name, {})
    yolo_dets = temporal.get("detections", [])
    yolo_match, yolo_iou = _best_match(bbox, yolo_dets, "box_xyxy")
    moge_match, moge_iou = _best_match(bbox, moge_by_image.get(frame_name, []), "box_xyxy")

    mast3r_center = None
    radar_surface = None
    temporal_center = temporal.get("pseudo_gt_vehicle_center_m")
    radar_depth_delta_existing = None
    if yolo_match:
        mast3r_center = yolo_match.get("pseudo_gt_vehicle_center_m")
        radar = yolo_match.get("radar_validation", {})
        radar_surface = radar.get("radar_surface_center_m")
        radar_depth_delta_existing = radar.get("radar_depth_delta_m")

    moge_center = None
    if moge_match:
        completion = moge_match.get("completion", {})
        moge_center = completion.get("aligned_completed_box_center_m")

    return {
        "yolo": {
            "best_iou": yolo_iou,
            "class": yolo_match.get("label") if yolo_match else None,
            "bbox_2d": yolo_match.get("box_xyxy") if yolo_match else None,
            "score": yolo_match.get("det_conf") if yolo_match else None,
        },
        "mast3r": {
            "center_3d": mast3r_center,
            "center_l2_delta_m": _dist3(center, mast3r_center),
            "depth_delta_m": _depth_delta(center, mast3r_center),
        },
        "moge_aligned": {
            "best_iou": moge_iou,
            "center_3d": moge_center,
            "center_l2_delta_m": _dist3(center, moge_center),
            "depth_delta_m": _depth_delta(center, moge_center),
        },
        "radar_weak_depth": {
            "surface_center_m": radar_surface,
            "mono_vs_radar_depth_delta_m": _depth_delta(center, radar_surface),
            "mast3r_vs_radar_depth_delta_m": radar_depth_delta_existing,
        },
        "temporal": {
            "frame_selected_center_3d": temporal_center,
            "center_l2_delta_m": _dist3(center, temporal_center),
            "depth_delta_m": _depth_delta(center, temporal_center),
            "frame_status": temporal.get("status"),
        },
    }


def _quality_flag(det: dict[str, Any]) -> str:
    cmp = det.get("comparison", {})
    if det.get("class") != "car":
        return "low"
    if cmp.get("yolo", {}).get("best_iou", 0.0) < 0.3:
        return "low"
    deltas = [
        cmp.get("mast3r", {}).get("depth_delta_m"),
        cmp.get("moge_aligned", {}).get("depth_delta_m"),
        cmp.get("temporal", {}).get("depth_delta_m"),
    ]
    finite = [x for x in deltas if isinstance(x, (int, float)) and math.isfinite(x)]
    if finite and min(finite) <= 5.0:
        return "medium"
    return "low"


def _prepare_resized_images(image_dir: Path, output_dir: Path, resize_hw: tuple[int, int] | None) -> tuple[Path, tuple[float, float]]:
    if resize_hw is None:
        return image_dir, (1.0, 1.0)
    target_h, target_w = resize_hw
    dst_dir = output_dir / f"monocon_resized_{target_w}x{target_h}"
    dst_dir.mkdir(parents=True, exist_ok=True)
    first_scale = None
    for src in sorted(image_dir.glob("*.jpg")):
        img = cv2.imread(str(src))
        if img is None:
            continue
        h, w = img.shape[:2]
        sx, sy = target_w / w, target_h / h
        if first_scale is None:
            first_scale = (sx, sy)
        resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(str(dst_dir / src.name), resized)
    return dst_dir, (first_scale or (1.0, 1.0))


def _unscale_bbox(box: list[float], scale_xy: tuple[float, float]) -> list[float]:
    sx, sy = scale_xy
    if sx == 1.0 and sy == 1.0:
        return box
    return [box[0] / sx, box[1] / sy, box[2] / sx, box[3] / sy]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--monocon-repo", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calib-json", type=Path, required=True)
    parser.add_argument("--temporal-json", type=Path, required=True)
    parser.add_argument("--moge-aligned-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resize-height", type=int, default=None)
    parser.add_argument("--resize-width", type=int, default=None)
    args = parser.parse_args()

    output_dir = args.output_json.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    resize_hw = None
    if args.resize_height and args.resize_width:
        resize_hw = (args.resize_height, args.resize_width)
    inference_image_dir, scale_xy = _prepare_resized_images(args.image_dir, output_dir, resize_hw)
    calib_txt = _make_kitti_raw_calib(args.calib_json, output_dir, scale_xy=scale_xy)

    sys.path.insert(0, str(args.monocon_repo))
    from dataset.kitti_raw_dataset import KITTIRawDataset
    from utils.engine_utils import move_data_device

    temporal_by_image = _index_temporal(_load_json(args.temporal_json))
    moge_by_image = _index_moge(_load_json(args.moge_aligned_json))

    dataset = KITTIRawDataset(str(inference_image_dir), str(calib_txt), img_extension="jpg")
    model = _load_monocon(args.monocon_repo, args.checkpoint, device, args.threshold)

    frames = []
    with torch.no_grad():
        for idx in range(len(dataset)):
            data = dataset[idx]
            image_path = data["img_metas"]["image_path"]
            if isinstance(image_path, list):
                image_path = image_path[0]
            frame_name = Path(image_path).name
            data = move_data_device(data, device)
            result = model.batch_eval(data, get_vis_format=True)[0]
            boxes_2d_by_class = result["img_bbox2d"]
            boxes_3d = result["img_bbox"]["boxes_3d"].numpy()
            scores = result["img_bbox"]["scores_3d"].numpy()
            labels = result["img_bbox"]["labels_3d"].numpy()

            detections = []
            class_offsets: dict[int, int] = {}
            for box3d, score, label in zip(boxes_3d, scores, labels):
                label_i = int(label)
                class_name = CLASSES[label_i] if 0 <= label_i < len(CLASSES) else str(label_i)
                class_boxes = boxes_2d_by_class[label_i] if 0 <= label_i < len(boxes_2d_by_class) else np.zeros((0, 5))
                class_det_i = class_offsets.get(label_i, 0)
                class_offsets[label_i] = class_det_i + 1
                bbox = _as_float_list(class_boxes[class_det_i, :4]) if class_det_i < len(class_boxes) else [None, None, None, None]
                if all(v is not None for v in bbox):
                    bbox = _unscale_bbox(bbox, scale_xy)
                center = _as_float_list(box3d[:3])
                det = {
                    "class": class_name,
                    "score": float(score),
                    "bbox_2d": bbox,
                    "center_3d": center,
                    "size_3d": _as_float_list(box3d[3:6]),
                    "yaw": float(box3d[6]),
                    "rotation_y": float(box3d[6]),
                    "depth": float(box3d[2]),
                    "camera_coordinate_convention": (
                        "KITTI camera coordinates inferred from MonoCon: x right, y down, z forward; "
                        "3D box center is shifted to bottom-center origin before export."
                    ),
                }
                if all(v is not None for v in bbox):
                    det["comparison"] = _build_comparison(
                        bbox, center, frame_name, temporal_by_image, moge_by_image
                    )
                    det["confidence_hint"] = _quality_flag(det)
                detections.append(det)

            frames.append({"image": frame_name, "detections": detections})

    payload = {
        "meta": {
            "model": "MonoCon",
            "implementation": "2gunsu/monocon-pytorch",
            "checkpoint": str(args.checkpoint),
            "images_dir": str(args.image_dir),
            "inference_images_dir": str(inference_image_dir),
            "resize_hw": list(resize_hw) if resize_hw else None,
            "input_scale_xy": list(scale_xy),
            "num_images": len(frames),
            "threshold": args.threshold,
            "device": device,
            "calib_source": str(args.calib_json),
            "generated_calib_file": str(calib_txt),
            "note": "Smoke test only; do not treat as final pseudo-GT until manual and cross-signal checks pass.",
        },
        "frames": frames,
    }

    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"wrote {args.output_json} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
