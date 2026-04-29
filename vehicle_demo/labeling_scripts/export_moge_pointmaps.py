import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import torch
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export MoGe/MoGe-2 metric point maps as per-image NPZ files.")
    p.add_argument("--images-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--pattern", type=str, default="*.jpg")
    p.add_argument("--model", type=str, default="Ruicheng/moge-2-vits-normal")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--max-side", type=int, default=1024, help="Resize longest image side before inference; 0 keeps original size.")
    p.add_argument("--resolution-level", type=int, default=5)
    p.add_argument("--num-tokens", type=int, default=0, help="Override MoGe-2 token count; 0 uses resolution-level.")
    p.add_argument("--fov-x", type=float, default=None)
    p.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"])
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def pick_device(name: str) -> torch.device:
    if name == "cuda":
        return torch.device("cuda")
    if name == "mps":
        return torch.device("mps")
    if name == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resize_for_inference(image_bgr: np.ndarray, max_side: int) -> np.ndarray:
    if max_side <= 0:
        return image_bgr
    h, w = image_bgr.shape[:2]
    scale = float(max_side) / float(max(h, w))
    if scale >= 1.0:
        return image_bgr
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def tensor_to_numpy(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().numpy()
    return np.asarray(value)


def cast_float(arr: Optional[np.ndarray], dtype: str) -> Optional[np.ndarray]:
    if arr is None:
        return None
    if arr.dtype == np.bool_:
        return arr
    if dtype == "float16" and np.issubdtype(arr.dtype, np.floating):
        return arr.astype(np.float16)
    if np.issubdtype(arr.dtype, np.floating):
        return arr.astype(np.float32)
    return arr


def main() -> None:
    args = parse_args()
    from moge.model.v2 import MoGeModel

    images = sorted(args.images_dir.glob(args.pattern))
    if args.limit > 0:
        images = images[: args.limit]
    if not images:
        raise RuntimeError(f"No images found in {args.images_dir} with pattern {args.pattern}")

    device = pick_device(args.device)
    use_fp16 = device.type in ("cuda", "mps")
    print(f"Loading {args.model} on {device}...")
    model = MoGeModel.from_pretrained(args.model).to(device).eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "task": "moge_pointmap_export",
        "model": args.model,
        "device": str(device),
        "source_images_dir": str(args.images_dir),
        "pattern": args.pattern,
        "max_side": int(args.max_side),
        "resolution_level": int(args.resolution_level),
        "num_tokens": None if args.num_tokens <= 0 else int(args.num_tokens),
        "dtype": args.dtype,
        "files": [],
    }

    for image_path in tqdm(images, desc="MoGe point maps"):
        out_path = args.output_dir / f"{image_path.stem}.npz"
        if out_path.exists() and not args.overwrite:
            manifest["files"].append({"image": str(image_path), "npz": str(out_path), "status": "exists"})
            continue

        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            manifest["files"].append({"image": str(image_path), "npz": str(out_path), "status": "read_error"})
            continue
        orig_h, orig_w = bgr.shape[:2]
        resized = resize_for_inference(bgr, args.max_side)
        inf_h, inf_w = resized.shape[:2]
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        image_t = torch.from_numpy(rgb).to(device=device, dtype=torch.float32).permute(2, 0, 1) / 255.0

        infer_kwargs: Dict = {
            "resolution_level": int(args.resolution_level),
            "fov_x": args.fov_x,
            "use_fp16": bool(use_fp16),
        }
        if args.num_tokens > 0:
            infer_kwargs["num_tokens"] = int(args.num_tokens)

        with torch.inference_mode():
            output = model.infer(image_t, **infer_kwargs)

        arrays = {}
        for key in ("points", "depth", "mask", "normal", "intrinsics"):
            arr = cast_float(tensor_to_numpy(output.get(key)), args.dtype)
            if arr is not None:
                arrays[key] = arr
        arrays["original_size_hw"] = np.asarray([orig_h, orig_w], dtype=np.int32)
        arrays["inference_size_hw"] = np.asarray([inf_h, inf_w], dtype=np.int32)
        np.savez_compressed(out_path, **arrays)
        manifest["files"].append(
            {
                "image": str(image_path),
                "npz": str(out_path),
                "status": "ok",
                "original_size_hw": [int(orig_h), int(orig_w)],
                "inference_size_hw": [int(inf_h), int(inf_w)],
                "keys": sorted(arrays.keys()),
            }
        )

    with (args.output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    ok = sum(1 for x in manifest["files"] if x["status"] in ("ok", "exists"))
    print(f"Saved {ok}/{len(manifest['files'])} point maps -> {args.output_dir}")


if __name__ == "__main__":
    main()
