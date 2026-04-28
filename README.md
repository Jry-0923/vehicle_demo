# vehicle_demo

Radar-camera fusion and image-dominant vehicle center labeling experiments.

This repository keeps the code, scripts, and notebooks needed to reproduce the workflow. Large local artifacts are intentionally not committed:

- raw datasets and frame folders (`3.13Test/`, `4.1Test/`, `4.15Test/`)
- generated outputs (`outputs/`)
- Python virtual environments (`venv/`, `mono3d_venv/`)
- model weights (`*.pt`, `*.pth`)
- packet captures (`*.pcapng`)

## Current Labeling Pipeline

The main research goal is to estimate per-frame per-vehicle 3D center labels with image geometry as the primary signal and radar only as weak validation.

Current pipeline:

1. Detect vehicles in images with YOLO boxes.
2. Estimate image-side 3D geometry with MASt3R point maps.
3. Estimate image-dominant vehicle center candidates inside each vehicle box.
4. Validate with BEV and temporal trajectory filtering.
5. Optionally compare against Mono3D-style outputs for A/B/C label grading.

Key scripts are copied into `vehicle_demo/labeling_scripts/` so they are tracked by this repository even though the local `vehicle_demo/ultralytics/` folder is an external checkout.

## Key Scripts

- `vehicle_demo/labeling_scripts/export_image_gt_centers.py`
- `vehicle_demo/labeling_scripts/filter_image_gt_tracks.py`
- `vehicle_demo/labeling_scripts/validate_image_gt_centers.py`
- `vehicle_demo/labeling_scripts/visualize_vehicle_centers.py`
- `vehicle_demo/labeling_scripts/compare_mono3d_centers.py`
- `vehicle_demo/labeling_scripts/prepare_mono3d_cloud_package.py`
- `vehicle_demo/notebooks/mono3d_colab_export.ipynb`

## Notes

Model weights should be downloaded separately. For example, YOLO weights can be downloaded from the Ultralytics release assets instead of being committed to Git.
