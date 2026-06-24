import argparse
import json
import re
import zipfile
from pathlib import Path
from typing import Dict, Optional

import cv2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Package images and metadata for cloud Mono3D inference.")
    p.add_argument("--images-dir", type=Path, required=True)
    p.add_argument("--temporal-json", type=Path, required=True)
    p.add_argument("--output-zip", type=Path, required=True)
    p.add_argument("--calib-json", type=Path, default=None)
    p.add_argument("--pattern", type=str, default="*.jpg")
    return p.parse_args()


def _timestamp_from_name(path: Path) -> Optional[float]:
    m = re.search(r"frame_(\d+)_(\d+)", path.stem)
    if not m:
        return None
    sec = int(m.group(1))
    frac = m.group(2)
    return float(f"{sec}.{frac}")


def _load_k(calib_json: Optional[Path]):
    if calib_json is None or not calib_json.exists():
        return None
    with calib_json.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    for key in ("K", "camera_matrix", "cam2img"):
        if key in obj:
            return obj[key]
    if "calibration" in obj:
        c = obj["calibration"]
        for key in ("K", "camera_matrix", "cam2img"):
            if key in c:
                return c[key]
    return None


def main() -> None:
    args = parse_args()
    images = sorted(args.images_dir.glob(args.pattern))
    if not images:
        raise RuntimeError(f"No images found in {args.images_dir}")
    with args.temporal_json.open("r", encoding="utf-8") as f:
        temporal = json.load(f)
    by_name: Dict[str, Dict] = {Path(fr.get("image", "")).name: fr for fr in temporal.get("frames", [])}
    cam2img = _load_k(args.calib_json)

    frames = []
    for idx, img_path in enumerate(images, start=1):
        img = cv2.imread(str(img_path))
        if img is None:
            raise RuntimeError(f"Cannot read image: {img_path}")
        h, w = img.shape[:2]
        fr = by_name.get(img_path.name, {})
        frames.append(
            {
                "frame_index": int(fr.get("frame_index", idx)),
                "file_name": img_path.name,
                "cloud_image_path": f"images/{img_path.name}",
                "original_path": str(img_path),
                "image_timestamp": fr.get("image_timestamp", _timestamp_from_name(img_path)),
                "width": int(w),
                "height": int(h),
                "cam2img": cam2img,
                "temporal_status": fr.get("status_temporal"),
                "temporal_selected_track_id": fr.get("selected_track_id"),
            }
        )

    manifest = {
        "meta": {
            "task": "mono3d_cloud_export",
            "source_images_dir": str(args.images_dir),
            "temporal_json": str(args.temporal_json),
            "calib_json": None if args.calib_json is None else str(args.calib_json),
            "num_frames": len(frames),
            "expected_output_schema": "outputs/mono3d_output_schema_example.json",
        },
        "frames": frames,
    }

    args.output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        zf.writestr("temporal_labels.json", json.dumps(temporal, ensure_ascii=False, indent=2))
        for img_path in images:
            zf.write(img_path, f"images/{img_path.name}")
    print(f"Packaged {len(frames)} images -> {args.output_zip}")


if __name__ == "__main__":
    main()
