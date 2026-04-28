#!/bin/zsh
set -euo pipefail

PY="/Users/jry/Research/vehicle_demo/venv/bin/python"
ROOT="/Users/jry/Research/vehicle_demo"
SCRIPT="$ROOT/ultralytics/scripts/export_vehicle_centers_mast3r.py"

# Optional local MASt3R checkpoint path.
# If empty, script will use HuggingFace model name (requires internet).
MAST3R_WEIGHTS="${MAST3R_WEIGHTS:-}"

run_one() {
  local target="$1"
  local radar_csv="$2"
  local calib_json="$3"
  local offset="$4"
  local output_json="$5"

  local cmd=(
    "$PY" "$SCRIPT"
    --images-dir "$ROOT/4.15Test/4_15_Frames/$target"
    --radar-csv "$radar_csv"
    --calib-json "$calib_json"
    --time-offset "$offset"
    --output-json "$output_json"
    --model "$ROOT/yolov8m.pt"
    --conf 0.05
    --imgsz 960
    --max-dt 0.2
    --match-shrink 0.0
    --match-min-points 1
    --mast3r-repo "$ROOT/third_party/mast3r"
    --mast3r-image-size 512
    --mast3r-device auto
    --mast3r-min-conf 0.5
    --mast3r-neighbor next
  )

  if [[ -n "$MAST3R_WEIGHTS" ]]; then
    cmd+=(--mast3r-weights "$MAST3R_WEIGHTS")
  fi

  echo "[MASt3R] Running $target -> $output_json"
  "${cmd[@]}"
}

run_one "Target1" "$ROOT/4.15Test/RawData/4.15雷达1_detection.csv" "$ROOT/outputs/calib_4.15_target1_auto.json" "-1776269908.253957" "$ROOT/outputs/4_15_target1_vehicle_centers_mast3r_frozen.json"
run_one "Target2" "$ROOT/4.15Test/RawData/4.15雷达2__detection.csv" "$ROOT/outputs/calib_4.15_target2_auto.json" "-1776269907.437394" "$ROOT/outputs/4_15_target2_vehicle_centers_mast3r_frozen.json"
run_one "Target3" "$ROOT/4.15Test/RawData/4.15雷达3_detection.csv" "$ROOT/outputs/calib_4.15_target3_auto.json" "-1776269906.80635" "$ROOT/outputs/4_15_target3_vehicle_centers_mast3r_frozen.json"
run_one "Target4" "$ROOT/4.15Test/RawData/4.15雷达4_detection.csv" "$ROOT/outputs/calib_4.15_target4_auto.json" "-1776269906.386734" "$ROOT/outputs/4_15_target4_vehicle_centers_mast3r_frozen.json"
run_one "Target5" "$ROOT/4.15Test/RawData/4.15雷达5_detection.csv" "$ROOT/outputs/calib_4.15_target5_auto.json" "-1776269906.847671" "$ROOT/outputs/4_15_target5_vehicle_centers_mast3r_frozen.json"
run_one "Target6" "$ROOT/4.15Test/RawData/4.15雷达6_detection.csv" "$ROOT/outputs/calib_4.15_target6_auto.json" "-1776269906.921231" "$ROOT/outputs/4_15_target6_vehicle_centers_mast3r_frozen.json"
run_one "Target7" "$ROOT/4.15Test/RawData/4.15雷达7_detection.csv" "$ROOT/outputs/calib_4.15_target7_auto.json" "-1776269907.765732" "$ROOT/outputs/4_15_target7_vehicle_centers_mast3r_frozen.json"

echo "[MASt3R] all targets done"
