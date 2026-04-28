#!/bin/zsh
set -euo pipefail

PY="/Users/jry/Research/vehicle_demo/venv/bin/python"
ROOT="/Users/jry/Research/vehicle_demo"

CALIB_JSON="$ROOT/outputs/calib_4.1_target7_auto.json"
TIME_OFFSET="-1775059104.290020"

$PY "$ROOT/ultralytics/scripts/export_vehicle_centers.py" \
  --images-dir "$ROOT/4.1Test/4_1_frames/Target7" \
  --radar-csv "$ROOT/4.1Test/RawData/4.1雷达数据7_detection.csv" \
  --use-object \
  --object-csv "$ROOT/4.1Test/RawData/4.1雷达数据7_object.csv" \
  --calib-json "$CALIB_JSON" \
  --time-offset "$TIME_OFFSET" \
  --disable-radar-only-fallback \
  --match-shrink 0.0 \
  --max-dt 0.2 \
  --output-json "$ROOT/outputs/4_1_target7_vehicle_centers_frozen.json" \
  --model "$ROOT/yolov8m.pt" \
  --conf 0.05
