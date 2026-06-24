#!/bin/zsh
set -euo pipefail

PY="/Users/jry/Research/vehicle_demo/venv/bin/python"
ROOT="/Users/jry/Research/vehicle_demo"

# Reuse the latest Target7 calibration as a starting point. Recalibrate if the
# camera/radar mounting changed between 4.1 and 4.15.
CALIB_JSON="$ROOT/outputs/calib_4.1_target7_auto.json"

$PY "$ROOT/ultralytics/scripts/export_vehicle_centers.py" \
  --images-dir "$ROOT/4.15Test/4_15_Frames/Target7" \
  --radar-csv "$ROOT/4.15Test/RawData/4.15雷达7_detection.csv" \
  --use-object \
  --object-csv "$ROOT/4.15Test/RawData/4.15雷达7_object.csv" \
  --calib-json "$CALIB_JSON" \
  --auto-offset \
  --disable-radar-only-fallback \
  --match-shrink 0.0 \
  --max-dt 0.2 \
  --output-json "$ROOT/outputs/4_15_target7_vehicle_centers_auto.json" \
  --model "$ROOT/yolov8m.pt" \
  --conf 0.05
