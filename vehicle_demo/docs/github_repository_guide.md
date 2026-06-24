# GeoVehicle3D GitHub Repository Guide

This repository should track source code, reproducible entrypoints, notebooks,
and small documentation artifacts only. It should not track local datasets,
model weights, generated outputs, virtual environments, or third-party
checkouts.

The GitHub main branch should present `GeoVehicle3D`: a vision-centric 3D
vehicle-center estimation pipeline. It should represent the research
contribution, not the
detector dependency. In this project, YOLO is a 2D proposal stage. The novel
part is the vehicle-center pseudo-labeling pipeline built around image geometry,
MASt3R/MoGe-style point maps, MonoDETR comparison, radar validation, temporal
filtering, and metric scale correction.

## Recommended Branch Layout

- `main`: clean public-facing branch for the vision-centric vehicle center
  estimation pipeline.
- `legacy/yolo-baseline`: old YOLO demo or detector-only baseline material.
- `legacy/radar-camera-fusion`: earlier radar-camera projection, calibration,
  and association experiments.
- feature branches: one branch per research direction, for example
  `feature/moge-completion`, `feature/monodetr-upgrade`, and
  `feature/anchor-scale`.

## What to Track

- `README.md`
- `.gitignore`
- `requirements.txt`
- `vehicle_demo/labeling_scripts/*.py`
- `vehicle_demo/scripts/*.sh`
- `vehicle_demo/docs/*.md`
- `vehicle_demo/docs/*.json` schema examples
- `vehicle_demo/notebooks/*.ipynb`
- legacy MATLAB helpers if they are still needed to explain raw radar parsing

## What to Keep Out of Git

- raw experiment folders such as `1.26Test/`, `3.13/`, `4.1/`, `4.15/`
- mirrored project datasets such as `vehicle_demo/4.15Test/`
- generated output folders such as `vehicle_demo/outputs/`
- model weights such as `*.pt`, `*.pth`, `*.onnx`, and `*.engine`
- local environments such as `vehicle_demo/venv/` and
  `vehicle_demo/mono3d_venv/`
- external repositories such as `vehicle_demo/third_party/mast3r`
- macOS metadata such as `.DS_Store`

## External Artifacts

Large files should be stored outside the Git repository. Recommended options:

- a cloud drive folder with the same experiment names used locally
- GitHub Releases for small, stable demo assets
- Git LFS only for a small number of necessary binary examples

For reproducibility, document artifact locations in a private note or release
description rather than committing the artifacts themselves.

## Current Dependency Boundary

`vehicle_demo/labeling_scripts/` contains the tracked copies of the active
research scripts. Some scripts import helper modules from a local Ultralytics
checkout under `vehicle_demo/ultralytics/scripts`. That folder is currently an
external checkout, not a clean Git submodule. For a public GitHub repository,
prefer one of these approaches:

1. Keep only the copied scripts under `vehicle_demo/labeling_scripts/` and
   document the external Ultralytics requirement.
2. Convert `vehicle_demo/ultralytics` into a real submodule with a valid
   `.gitmodules` entry.
3. Vendor only the minimal helper modules needed by the labeling scripts.

The first option is the least disruptive for the current checkout.

## Cleanup Commands

To remove already-tracked local artifacts from Git while keeping the files on
disk:

```bash
git rm -r --cached --ignore-unmatch .DS_Store
git rm -r --cached --ignore-unmatch 1.26Test 3.13 4.1 4.15
git rm -r --cached --ignore-unmatch vehicle_demo/3.13Test vehicle_demo/4.1Test vehicle_demo/4.15Test
git rm -r --cached --ignore-unmatch vehicle_demo/outputs
git rm -r --cached --ignore-unmatch vehicle_demo/venv vehicle_demo/mono3d_venv
git rm -r --cached --ignore-unmatch '*.zip' '*.pcapng' '*.pt' '*.pth' '*.onnx' '*.engine' '*.mat'
```

After cleanup, verify the index with:

```bash
git status --short
git ls-files | wc -l
```
