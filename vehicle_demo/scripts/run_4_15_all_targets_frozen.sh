#!/bin/zsh
set -euo pipefail

PY="/Users/jry/Research/vehicle_demo/venv/bin/python"
ROOT="/Users/jry/Research/vehicle_demo"
MODEL="$ROOT/yolov8m.pt"
SCRIPT="$ROOT/ultralytics/scripts/export_vehicle_centers.py"

run_one() {
  local n="$1"
  local det_csv="$2"
  local obj_csv="$3"
  local calib_json="$4"
  local time_offset="$5"

  $PY "$SCRIPT" \
    --images-dir "$ROOT/4.15Test/4_15_Frames/Target${n}" \
    --radar-csv "$det_csv" \
    --use-object \
    --object-csv "$obj_csv" \
    --calib-json "$calib_json" \
    --time-offset "$time_offset" \
    --disable-radar-only-fallback \
    --match-shrink 0.0 \
    --max-dt 0.2 \
    --output-json "$ROOT/outputs/4_15_target${n}_vehicle_centers_frozen.json" \
    --model "$MODEL" \
    --conf 0.05
}

run_one 1 "$ROOT/4.15Test/RawData/4.15雷达1_detection.csv" "$ROOT/4.15Test/RawData/4.15雷达1_object.csv" "$ROOT/outputs/calib_4.15_target1_auto.json" "-1776269908.253957"
run_one 2 "$ROOT/4.15Test/RawData/4.15雷达2__detection.csv" "$ROOT/4.15Test/RawData/4.15雷达2__object.csv" "$ROOT/outputs/calib_4.15_target2_auto.json" "-1776269907.437394"
run_one 3 "$ROOT/4.15Test/RawData/4.15雷达3_detection.csv" "$ROOT/4.15Test/RawData/4.15雷达3_object.csv" "$ROOT/outputs/calib_4.15_target3_auto.json" "-1776269906.806350"
run_one 4 "$ROOT/4.15Test/RawData/4.15雷达4_detection.csv" "$ROOT/4.15Test/RawData/4.15雷达4_object.csv" "$ROOT/outputs/calib_4.15_target4_auto.json" "-1776269906.386734"
run_one 5 "$ROOT/4.15Test/RawData/4.15雷达5_detection.csv" "$ROOT/4.15Test/RawData/4.15雷达5_object.csv" "$ROOT/outputs/calib_4.15_target5_auto.json" "-1776269906.847671"
run_one 6 "$ROOT/4.15Test/RawData/4.15雷达6_detection.csv" "$ROOT/4.15Test/RawData/4.15雷达6_object.csv" "$ROOT/outputs/calib_4.15_target6_auto.json" "-1776269906.921231"
run_one 7 "$ROOT/4.15Test/RawData/4.15雷达7_detection.csv" "$ROOT/4.15Test/RawData/4.15雷达7_object.csv" "$ROOT/outputs/calib_4.15_target7_auto.json" "-1776269907.765732"

echo "Done all targets. Summary: $ROOT/outputs/4_15_all_targets_summary.json"
