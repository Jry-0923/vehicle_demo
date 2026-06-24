#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
ROOT="${SCRIPT_DIR:h}"
PY="${PY:-$ROOT/venv/bin/python}"
SCRIPT="$ROOT/labeling_scripts/export_image_gt_centers.py"

# Pure vision export:
# - YOLO finds vehicle boxes.
# - MASt3R estimates image-side 3D geometry inside each vehicle box.
# - Radar CSVs are not loaded and are not used for matching, filtering, or confidence.
MAST3R_WEIGHTS="${MAST3R_WEIGHTS:-}"

run_one() {
  local target_num="$1"
  local target="Target${target_num}"
  local calib_json="$ROOT/outputs/calib_4.15_target${target_num}_auto.json"
  local output_json="$ROOT/outputs/4_15_target${target_num}_visual_only_centers.json"

  local cmd=(
    "$PY" "$SCRIPT"
    --visual-only
    --images-dir "$ROOT/4.15Test/4_15_Frames/$target"
    --calib-json "$calib_json"
    --output-json "$output_json"
    --model "$ROOT/yolov8m.pt"
    --conf 0.05
    --imgsz 960
    --box-inner-scale 0.72
    --depth-trim 0.15
    --min-image-points 80
    --mast3r-repo "$ROOT/third_party/mast3r"
    --mast3r-image-size 512
    --mast3r-device auto
    --mast3r-min-conf 0.5
    --mast3r-neighbor next
  )

  if [[ -n "$MAST3R_WEIGHTS" ]]; then
    cmd+=(--mast3r-weights "$MAST3R_WEIGHTS")
  fi

  echo "[visual-only] Running $target -> $output_json"
  "${cmd[@]}"
}

for target_num in 1 2 3 4 5 6 7; do
  run_one "$target_num"
done

echo "[visual-only] all targets done"
