# GeoVehicle3D

Vision-centric 3D vehicle-center estimation from image geometry.

This repository contains a research pipeline for estimating vehicle centers from
image geometry. YOLO is used only as a 2D vehicle proposal generator; the core
project contribution is the downstream 3D center pseudo-labeling and validation
pipeline.

## Project Focus

`GeoVehicle3D` should be presented as a vision-centric vehicle-center
estimation system, not as a YOLO detection demo.

The project-specific work is:

- image-dominant 3D center pseudo-label generation from vehicle boxes and
  MASt3R point maps
- radar used as weak validation and calibration context, not as the main label
  source
- temporal and BEV filtering for label stability
- MonoDETR / Mono3D comparison for label grading and failure analysis
- MoGe-lite visible-surface completion as a geometry refinement path
- fixed-scene-anchor scale correction for metric consistency

## Pipeline

1. Detect vehicles in image frames with YOLO.
2. Estimate image-side 3D geometry with MASt3R point maps.
3. Build per-vehicle center candidates inside each 2D vehicle box.
4. Validate labels with radar depth checks, BEV checks, temporal filtering, and
   optional Mono3D-style comparisons.
5. Export JSON labels for downstream review or training experiments.

Large datasets, generated outputs, model weights, virtual environments, and
third-party checkouts are intentionally excluded from Git. Keep those files
locally or in external storage.

## Branch Intent

- `main`: clean, lightweight project branch centered on the novel
  vision-geometry vehicle-center pipeline.
- `yolo-baseline` or `legacy/yolo-baseline`: older YOLO detection demo and
  detector-only baseline material.
- `radar-camera-fusion-legacy`: earlier radar-camera fusion and calibration
  experiments used as context for the current vision-centric work.

## Repository Layout

```text
.
├── README.md
├── requirements.txt
├── vehicle_demo/
│   ├── labeling_scripts/      # tracked research scripts used for label export and review
│   ├── scripts/               # reproducible shell entrypoints for experiments
│   ├── docs/                  # workflow notes and output schema examples
│   ├── notebooks/             # notebook-based export or cloud workflow helpers
│   ├── outputs/               # ignored generated artifacts
│   ├── models/                # ignored model weights
│   ├── third_party/           # ignored external checkouts such as MASt3R
│   ├── venv/                  # ignored local Python environment
│   └── mono3d_venv/           # ignored local Mono3D/OpenMMLab environment
├── 1.26Test/, 3.13/, 4.1/, 4.15/
│   └── ignored local datasets
└── RadarDataAnalysis.m, RadarRawDataProcess.m, Calculator.m
    └── legacy MATLAB radar-processing helpers
```

See `vehicle_demo/docs/github_repository_guide.md` for the intended GitHub
cleanup policy and artifact-handling rules.

## Main Entry Points

- `vehicle_demo/labeling_scripts/export_image_gt_centers.py` exports
  image-dominant vehicle center labels. Use `--visual-only` to run without
  radar CSV input.
- `vehicle_demo/scripts/run_4_15_all_targets_visual_only.sh` runs the
  visual-only pipeline for all 4.15 targets.
- `vehicle_demo/labeling_scripts/filter_image_gt_tracks.py` applies temporal
  filtering to exported labels.
- `vehicle_demo/labeling_scripts/validate_image_gt_centers.py` validates label
  quality and emits review summaries.
- `vehicle_demo/labeling_scripts/compare_mono3d_centers.py` compares exported
  labels with Mono3D-style results.
- `vehicle_demo/labeling_scripts/upgrade_labels_with_monodetr.py` upgrades label
  grades using MonoDETR outputs.
- `vehicle_demo/labeling_scripts/estimate_anchor_scale.py` and
  `vehicle_demo/labeling_scripts/apply_anchor_scale_to_labels.py` support the
  fixed-anchor scale-correction workflow.

## Setup

Create a local environment outside Git tracking:

```bash
cd /Users/jry/Radar_Camera_Research
python3 -m venv vehicle_demo/venv
vehicle_demo/venv/bin/pip install -r requirements.txt
```

Some workflows require additional external repositories or model weights:

- Ultralytics / YOLO weights such as `yolov8m.pt`
- MASt3R under `vehicle_demo/third_party/mast3r`
- MoGe or MonoDETR environments for optional pure-vision refinement
- OpenMMLab / MMDetection3D in a separate environment for Mono3D baselines

Do not commit these dependencies or generated artifacts.

## Example

```bash
cd /Users/jry/Radar_Camera_Research/vehicle_demo
PY=venv/bin/python scripts/run_4_15_all_targets_visual_only.sh
```

For the anchor-scale workflow, see
`vehicle_demo/docs/anchor_scale_workflow.md`.

For MoGe-lite completion, see `vehicle_demo/docs/moge_lite_notes.md`.

For Mono3D integration notes, see
`vehicle_demo/docs/mono3d_next_step_notes.md`.
