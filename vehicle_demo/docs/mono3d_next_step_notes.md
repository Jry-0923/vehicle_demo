# Mono3D 接入下一步

当前本地 venv 没有 `mmdet3d/mmcv/mmdet/mmengine`，所以没有直接在现有环境里跑 SMOKE。不要直接把 OpenMMLab 装进当前 venv，避免破坏已经跑通的 YOLO + MASt3R 环境。

建议做法：

1. 新建独立环境安装 MMDetection3D/SMOKE。
2. 用 KITTI 预训练 SMOKE/FCOS3D/PGD 跑 `4.15Test/4_15_Frames/Target7`。
3. 把结果导出为 `mono3d_output_schema_example.json` 这种格式。
4. 回到本项目运行：

```bash
/Users/jry/Radar_Camera_Research/vehicle_demo/venv/bin/python   /Users/jry/Radar_Camera_Research/vehicle_demo/ultralytics/scripts/compare_mono3d_centers.py   --temporal-json /Users/jry/Radar_Camera_Research/vehicle_demo/outputs/4_15_target7_image_gt_centers_temporal.json   --mono3d-json /path/to/mono3d_target7_outputs.json   --output-json /Users/jry/Radar_Camera_Research/vehicle_demo/outputs/4_15_target7_final_labels_mono3d.json   --summary-json /Users/jry/Radar_Camera_Research/vehicle_demo/outputs/4_15_target7_final_labels_mono3d_summary.json
```

分级逻辑：

- A 级：MASt3R 时序中心和 Mono3D 中心距离 <= 1.0m。
- B 级：MASt3R 时序中心稳定，但 Mono3D 缺失或距离 <= 2.0m。
- C 级：两者差异 > 2.0m，不建议训练用。
