# MoGe-lite vehicle center completion

This route treats MoGe/MoGe-2 as a monocular metric geometry backend, not as a
3D box detector. A vehicle box contains only the visible surface point cloud, so
the visible point centroid should not be used directly as the vehicle center.

The intended workflow is:

1. Run the existing image-dominant detector to get per-frame vehicle boxes.
2. Run MoGe/MoGe-2 and save one `.npz` file per image with a metric point map
   or metric depth plus intrinsics.
3. Use `moge_box_completion_validator.py` to sample visible vehicle points
   inside each 2D box.
4. Fit a BEV-oriented footprint to the visible surface.
5. Complete the unseen part with vehicle class size priors, then export:
   - `visible_surface_center_m`
   - `completed_box_center_m`
   - `completed_box_bev_corners_xz`
   - `completion_confidence`

This is a validation and pseudo-label refinement step. The confidence should be
lower for extreme one-face views, heavy occlusion, or when most of the vehicle
volume is completed from priors.

Export MoGe point maps first:

```bash
python vehicle_demo/labeling_scripts/export_moge_pointmaps.py \
  --images-dir vehicle_demo/4.15Test/4_15_Frames/Target7 \
  --output-dir vehicle_demo/outputs/moge_target7_npz \
  --max-side 768 \
  --resolution-level 5 \
  --dtype float16
```

Then run box completion:

```bash
python vehicle_demo/labeling_scripts/moge_box_completion_validator.py \
  --centers-json vehicle_demo/outputs/4_15_target7_image_gt_centers_temporal.json \
  --moge-dir vehicle_demo/outputs/moge_target7_npz \
  --output-json vehicle_demo/outputs/4_15_target7_moge_lite_completion.json \
  --bev-png vehicle_demo/outputs/4_15_target7_moge_lite_completion_bev.png
```

The script automatically rescales original-image `box_xyxy` coordinates to the MoGe point-map resolution when the source image is available.

Expected MoGe `.npz` keys are flexible. The script auto-detects common names:

- point map: `points`, `pointmap`, `pts3d`, `points3d`, `xyz`
- depth: `depth`, `metric_depth`, `depth_m`
- intrinsics: `intrinsics`, `K`, `cam2img`, `camera_matrix`
- mask/confidence: `mask`, `valid_mask`, `confidence`, `conf`

If only depth is stored, the `.npz` must also contain a 3x3 intrinsics matrix so
the script can back-project pixels into metric 3D points.

Align MoGe coordinates to the current project pseudo-label frame:

```bash
python vehicle_demo/labeling_scripts/align_moge_completion.py \
  --moge-completion-json vehicle_demo/outputs/4_15_target7_moge_lite_completion.json \
  --output-json vehicle_demo/outputs/4_15_target7_moge_lite_completion_aligned.json \
  --bev-png vehicle_demo/outputs/4_15_target7_moge_lite_completion_aligned_bev.png \
  --mode uniform \
  --fit-source visible
```

Use `--mode affine` only as a diagnostic. It can reduce fit error, but it is less physically constrained than uniform scale plus translation.
