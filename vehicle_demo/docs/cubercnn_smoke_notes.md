# Cube R-CNN smoke test notes

Smoke-test target:

- Input: first 5 images from `4.15Test/4_15_Frames/Target7`
- Model: Cube R-CNN DLA34, Omni3D pretrained checkpoint
- Output JSON: `outputs/cubercnn_target7_smoke5.json`
- Additional diagnostic JSON: `outputs/cubercnn_target7_smoke5_allclasses.json`

Result summary:

- Vehicle-filtered output found 4/5 detections, but the boxes were on road
  markings rather than the visible car.
- All-class output did detect the visible right-side vehicle with high 2D IoU
  against the existing YOLO car box, but Cube R-CNN classified it as `table`
  instead of `car`.
- The matched Cube detection had plausible vehicle-like dimensions
  (`length ~= 4.5m`, `width ~= 1.9m`, `height ~= 1.6m`), but its depth was
  about `38m`, while the current MASt3R/temporal pseudo-label depth was about
  `15m`.

Decision:

Do not batch-run Cube R-CNN on all 109 frames yet. In this scene it is useful as
a possible 3D geometry prior only after YOLO matching, but it fails the planned
smoke-test gates:

- Vehicle class recognition: failed (`table` for the actual car).
- 2D box alignment: partially passed in all-class mode, failed in vehicle-only
  mode.
- 3D center depth: failed relative to the existing image-dominant pseudo-labels.
- Size/yaw: size looked vehicle-like for the all-class matched box, but the
  class/depth conflict makes the label low confidence.

Next branch:

Move to MonoCon or SMOKE independent implementations. Keep Cube R-CNN results as
a diagnostic baseline, not as a pseudo-GT source for this dataset.

Commands used:

```bash
python vehicle_demo/labeling_scripts/export_cubercnn_smoke.py \
  --input-folder /Users/jry/Radar_Camera_Research/vehicle_demo/outputs/cubercnn_smoke_input \
  --output-json /Users/jry/Radar_Camera_Research/vehicle_demo/outputs/cubercnn_target7_smoke5.json \
  --config-file /Users/jry/Radar_Camera_Research/vehicle_demo/third_party/omni3d/configs/cubercnn_DLA34_FPN.yaml \
  --weights /Users/jry/Radar_Camera_Research/vehicle_demo/models/cubercnn/cubercnn_DLA34_FPN.pth \
  --calib-json /Users/jry/Radar_Camera_Research/vehicle_demo/outputs/calib_4.15_target7_auto.json \
  --threshold 0.25 \
  --device cpu
```

```bash
python vehicle_demo/labeling_scripts/export_cubercnn_smoke.py \
  --input-folder /Users/jry/Radar_Camera_Research/vehicle_demo/outputs/cubercnn_smoke_input \
  --output-json /Users/jry/Radar_Camera_Research/vehicle_demo/outputs/cubercnn_target7_smoke5_allclasses.json \
  --config-file /Users/jry/Radar_Camera_Research/vehicle_demo/third_party/omni3d/configs/cubercnn_DLA34_FPN.yaml \
  --weights /Users/jry/Radar_Camera_Research/vehicle_demo/models/cubercnn/cubercnn_DLA34_FPN.pth \
  --calib-json /Users/jry/Radar_Camera_Research/vehicle_demo/outputs/calib_4.15_target7_auto.json \
  --threshold 0.15 \
  --device cpu \
  --include-nonvehicles
```
