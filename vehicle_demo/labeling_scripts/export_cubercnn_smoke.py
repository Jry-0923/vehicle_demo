import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch


VEHICLE_CLASSES = {"car", "truck", "bus", "van", "trailer"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Cube R-CNN/Omni3D inference and export compact vehicle 3D boxes.")
    p.add_argument("--omni3d-repo", type=Path, default=Path("vehicle_demo/third_party/omni3d"))
    p.add_argument("--input-folder", type=Path, required=True)
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--config-file", type=str, default="cubercnn://omni3d/cubercnn_DLA34_FPN.yaml")
    p.add_argument("--weights", type=str, default="cubercnn://omni3d/cubercnn_DLA34_FPN.pth")
    p.add_argument("--threshold", type=float, default=0.25)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--focal-length", type=float, default=0.0)
    p.add_argument("--principal-point", type=float, nargs=2, default=None)
    p.add_argument("--calib-json", type=Path, default=None)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--include-nonvehicles", action="store_true")
    return p.parse_args()


def _pick_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    return name


def _load_calib(calib_json: Optional[Path]) -> Optional[np.ndarray]:
    if calib_json is None or not calib_json.exists():
        return None
    obj = json.load(open(calib_json, "r", encoding="utf-8"))
    for key in ("K", "camera_matrix", "cam2img"):
        if key in obj:
            return np.asarray(obj[key], dtype=np.float32)
    if "calibration" in obj:
        for key in ("K", "camera_matrix", "cam2img"):
            if key in obj["calibration"]:
                return np.asarray(obj["calibration"][key], dtype=np.float32)
    return None


def _intrinsics(args: argparse.Namespace, width: int, height: int) -> np.ndarray:
    K0 = _load_calib(args.calib_json)
    if K0 is not None and K0.shape == (3, 3):
        return K0
    focal = float(args.focal_length)
    if focal <= 0:
        focal = 4.0 * height / 2.0
    if args.principal_point is None:
        px, py = width / 2.0, height / 2.0
    else:
        px, py = args.principal_point
    return np.asarray([[focal, 0.0, px], [0.0, focal, py], [0.0, 0.0, 1.0]], dtype=np.float32)


def _yaw_from_pose(pose: np.ndarray) -> Optional[float]:
    if pose.shape == (3, 3):
        return float(math.atan2(float(pose[0, 2]), float(pose[2, 2])))
    return None


def _to_list(x) -> List[float]:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return [float(v) for v in np.asarray(x).reshape(-1).tolist()]


def _load_categories(config_file: str, util) -> List[str]:
    category_path = os.path.join(util.file_parts(config_file)[0], "category_meta.json")
    if category_path.startswith(util.CubeRCNNHandler.PREFIX):
        category_path = util.CubeRCNNHandler._get_local_path(util.CubeRCNNHandler, category_path)
    if os.path.exists(category_path):
        return util.load_json(category_path)["thing_classes"]
    return [
        "chair", "table", "cabinet", "car", "lamp", "books", "sofa", "pedestrian", "picture", "window",
        "pillow", "truck", "door", "blinds", "sink", "shelves", "television", "shoes", "cup", "bottle",
        "bookcase", "laptop", "desk", "cereal box", "floor mat", "traffic cone", "mirror", "barrier",
        "counter", "camera", "bicycle", "toilet", "bus", "bed", "refrigerator", "trailer", "box", "oven",
        "clothes", "van", "towel", "motorcycle", "night stand", "stove", "machine", "stationery", "bathtub",
        "cyclist", "curtain", "bin",
    ]


def main() -> None:
    args = parse_args()
    repo = args.omni3d_repo.resolve()
    if not repo.exists():
        raise FileNotFoundError(f"Omni3D repo not found: {repo}")
    sys.path.insert(0, str(repo))
    os.chdir(repo)

    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.config import get_cfg
    from detectron2.data import transforms as T
    from cubercnn.config import get_cfg_defaults
    from cubercnn.modeling.proposal_generator import RPNWithIgnore  # noqa: F401
    from cubercnn.modeling.roi_heads import ROIHeads3D  # noqa: F401
    from cubercnn.modeling.meta_arch import RCNN3D, build_model  # noqa: F401
    from cubercnn.modeling.backbone import build_dla_from_vision_fpn_backbone  # noqa: F401
    from cubercnn import util

    device = _pick_device(args.device)
    cfg = get_cfg()
    get_cfg_defaults(cfg)
    config_file = args.config_file
    if config_file.startswith(util.CubeRCNNHandler.PREFIX):
        config_file = util.CubeRCNNHandler._get_local_path(util.CubeRCNNHandler, config_file)
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(["MODEL.WEIGHTS", args.weights, "MODEL.DEVICE", device, "OUTPUT_DIR", str(args.output_json.parent)])
    cfg.freeze()

    model = build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=True)
    model.eval()

    cats = _load_categories(args.config_file, util)
    images = sorted([p for p in args.input_folder.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if args.limit > 0:
        images = images[: args.limit]
    min_size = cfg.INPUT.MIN_SIZE_TEST
    max_size = cfg.INPUT.MAX_SIZE_TEST
    augmentations = T.AugmentationList([T.ResizeShortestEdge(min_size, max_size, "choice")])

    payload = {
        "meta": {
            "backend": "Cube R-CNN / Omni3D",
            "config_file": args.config_file,
            "weights": args.weights,
            "threshold": float(args.threshold),
            "device": device,
            "note": "Cube R-CNN pred_dimensions are WHL internally; size_3d is exported as [length, width, height].",
        },
        "frames": [],
    }

    with torch.no_grad():
        for image_path in images:
            im = cv2.imread(str(image_path))
            if im is None:
                payload["frames"].append({"image": str(image_path), "status": "read_error", "detections": []})
                continue
            h, w = im.shape[:2]
            K = _intrinsics(args, w, h)
            aug_input = T.AugInput(im)
            _ = augmentations(aug_input)
            image = aug_input.image
            image_t = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)), device=device)
            batched = [{"image": image_t, "height": h, "width": w, "K": K}]
            inst = model(batched)[0]["instances"].to("cpu")
            detections = []
            n = len(inst)
            for i in range(n):
                score = float(inst.scores[i])
                if score < args.threshold:
                    continue
                cls_idx = int(inst.pred_classes[i])
                cls_name = cats[cls_idx] if 0 <= cls_idx < len(cats) else str(cls_idx)
                if not args.include_nonvehicles and cls_name not in VEHICLE_CLASSES:
                    continue
                whl = np.asarray(_to_list(inst.pred_dimensions[i]), dtype=np.float32)
                size_lwh = [float(whl[2]), float(whl[0]), float(whl[1])] if len(whl) == 3 else _to_list(whl)
                pose = inst.pred_pose[i].detach().cpu().numpy()
                det = {
                    "class": cls_name,
                    "score": score,
                    "bbox_2d": _to_list(inst.pred_boxes.tensor[i]),
                    "center_3d": _to_list(inst.pred_center_cam[i]),
                    "size_3d": size_lwh,
                    "yaw": _yaw_from_pose(pose),
                    "cubercnn_dimensions_whl": _to_list(whl),
                    "center_2d": _to_list(inst.pred_center_2D[i]),
                }
                detections.append(det)
            payload["frames"].append(
                {
                    "image": str(image_path),
                    "width": int(w),
                    "height": int(h),
                    "K": K.tolist(),
                    "detections": detections,
                }
            )
            print(f"{image_path.name}: {len(detections)} vehicle detections")

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
