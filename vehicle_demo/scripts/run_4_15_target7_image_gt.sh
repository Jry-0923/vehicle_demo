#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
ROOT="${SCRIPT_DIR:h}"
PY="$ROOT/venv/bin/python"

# Image-dominant pseudo-GT export:
# - YOLO finds vehicle boxes.
# - MASt3R estimates image-side 3D geometry inside each vehicle box.
# - Radar is kept only as a weak depth validation signal, not as the center source.
CALIB_JSON="$ROOT/outputs/calib_4.15_target7_auto.json"
if [[ ! -f "$CALIB_JSON" ]]; then
  CALIB_JSON="$ROOT/outputs/calib_4.1_target7_auto.json"
fi

$PY "$ROOT/ultralytics/scripts/export_image_gt_centers.py" \
  --images-dir "$ROOT/4.15Test/4_15_Frames/Target7" \
  --radar-csv "$ROOT/4.15Test/RawData/4.15雷达7_detection.csv" \
  --calib-json "$CALIB_JSON" \
  --auto-offset \
  --match-shrink 0.0 \
  --max-dt 0.2 \
  --output-json "$ROOT/outputs/4_15_target7_image_gt_centers.json" \
  --model "$ROOT/yolov8m.pt" \
  --conf 0.05 \
  --box-inner-scale 0.72 \
  --depth-trim 0.15 \
  --min-image-points 80 \
  --radar-depth-tolerance-m 2.0
