# Anchor Scale Workflow

Use fixed-height scene anchors to estimate a global metric scale correction for
MASt3R camera-frame 3D centers.

Known anchors:

- railing vertical height: `0.92m`

Do not use curb anchors for this dataset. The curb location is not reliable in
the images, so the scale workflow treats the railing as the only physical
height anchor.

## 1. Prepare anchor review images

```bash
cd /Users/jry/Radar_Camera_Research/vehicle_demo
venv/bin/python labeling_scripts/prepare_anchor_scale_review.py
```

This writes:

- `outputs/anchor_scale_review/images/`
- `outputs/anchor_scale_review/anchors.csv`

Fill `anchors.csv` with original image pixel coordinates:

- `bottom_u,bottom_v`
- `top_u,top_v`
- `true_height_m`

Leave unusable rows blank.

The easiest way to fill the pixel coordinates is the click labeler:

```bash
cd /Users/jry/Radar_Camera_Research/vehicle_demo
venv/bin/python labeling_scripts/label_anchor_points.py \
  --anchors-csv outputs/anchor_scale_review/anchors.csv \
  --anchor-type railing
```

For each row, click:

1. bottom endpoint
2. top endpoint
3. press Enter or Space to save

Keyboard controls:

- `u`: undo the last click
- `s`: skip this anchor row
- `q`: save current CSV and quit

The labeler should be run on railing rows only:

```bash
venv/bin/python labeling_scripts/label_anchor_points.py \
  --anchors-csv outputs/anchor_scale_review/anchors.csv \
  --anchor-type railing
```

## 2. Estimate anchor scale

```bash
cd /Users/jry/Radar_Camera_Research/vehicle_demo
venv/bin/python labeling_scripts/estimate_anchor_scale.py \
  --anchors-csv outputs/anchor_scale_review/anchors.csv \
  --output-json outputs/anchor_scale_review/anchor_scale_summary.json \
  --output-csv outputs/anchor_scale_review/anchor_scale_measurements.csv
```

Key outputs:

- `scale_median`
- `scale_std`
- `bad_anchor_outliers`

## 3. Apply scale to labels

```bash
cd /Users/jry/Radar_Camera_Research/vehicle_demo
venv/bin/python labeling_scripts/apply_anchor_scale_to_labels.py \
  --scale-json outputs/anchor_scale_review/anchor_scale_summary.json \
  --input-dir outputs/monodetr_upgraded_50m_all_strong_only \
  --output-dir outputs/anchor_scaled_labels
```

This preserves original center fields and writes `scale_corrected_*` fields.
Image projections do not change under uniform scale.

## 4. Compare on the gold set

```bash
cd /Users/jry/Radar_Camera_Research/vehicle_demo
venv/bin/python labeling_scripts/compare_gold_set_center_versions.py \
  --scaled-dir outputs/anchor_scaled_labels \
  --output-csv outputs/anchor_scale_review/gold_set_center_comparison.csv \
  --summary-json outputs/anchor_scale_review/gold_set_center_comparison_summary.json
```

If a box-completion JSON is available later, pass it with `--completed-json`.
