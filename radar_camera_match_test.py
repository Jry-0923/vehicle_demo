import numpy as np
import pandas as pd
import cv2
import os
import re
import glob
from ultralytics import YOLO

# ==========================================
# ⚙️ 1. 核心参数配置
# ==========================================
K = np.array([
    [2800.122005, 0, 1273.485366],
    [0, 2799.104268, 783.562700],
    [0, 0, 1.0]
])
dist_coeffs = np.array([-0.52425, 0.62195, -0.00073745, 0.0017935, 0.12563])

R_original = np.array([
    [0.9564, 0, -0.2921],
    [0, 1.0000, 0],
    [0.2921, 0, 0.9564]
])

# -----------------------------------------------------------
# 调试控制台
# -----------------------------------------------------------
# 【旋转修正 - 单位：度】
yaw_correction_deg   = -1.05   # 左右偏航 (绕 Z 轴)
pitch_correction_deg = +1.42   # 上下俯仰 (绕 X 轴)
roll_correction_deg  = -5.15   # 左右翻滚 (绕 Y 轴)

# 【平移修正 - 单位：米】
tx_correction_m = 0.0        # X 方向偏移 (前后)
ty_correction_m = -0.0        # Y 方向偏移 (左右)
tz_correction_m = +4.32        # Z 方向偏移 (上下)

# --- 旋转矩阵计算 ---
y_rad = np.deg2rad(yaw_correction_deg)
p_rad = np.deg2rad(pitch_correction_deg)
r_rad = np.deg2rad(roll_correction_deg)

R_yaw = np.array([
    [np.cos(y_rad), -np.sin(y_rad), 0],
    [np.sin(y_rad),  np.cos(y_rad), 0],
    [0,              0,             1]
])
R_pitch = np.array([
    [1, 0, 0],
    [0, np.cos(p_rad), -np.sin(p_rad)],
    [0, np.sin(p_rad),  np.cos(p_rad)]
])
R_roll = np.array([
    [np.cos(r_rad),  0, np.sin(r_rad)],
    [0,              1, 0],
    [-np.sin(r_rad), 0, np.cos(r_rad)]
])

# 合成最终旋转矩阵
R_final = np.dot(R_original, np.dot(R_yaw, np.dot(R_pitch, R_roll)))

# --- 平移向量计算 ---
t_original = np.array([[12.0], [40.0], [-18.0]]) / 100.0
t_final = t_original + np.array([[tx_correction_m], [ty_correction_m], [tz_correction_m]])

# 传递给算法使用
R = R_final
t = t_final

# 打印修正后的外参
print("\n" + "="*30)
print("修正后的外参输出 (Extrinsics):")
print("-" * 30)
print(f"Rotation Matrix (R):\n{R}")
print("-" * 30)
print(f"Translation Vector (t):\n{t}")
print("="*30 + "\n")

# ==========================================
# 全局变量与鼠标回调函数
# ==========================================
current_frame_points = []

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("\n" + "="*40)
        print(f"点击像素位置 (u, v): ({x}, {y})")
        hit_radar_point = None
        min_dist = 12.0  
        for pt in current_frame_points:
            dist = np.sqrt((pt['u'] - x)**2 + (pt['v'] - y)**2)
            if dist < min_dist:
                hit_radar_point = pt
                min_dist = dist 
        if hit_radar_point:
            print(f"命中散点")
            print(f"   物理空间坐标 (x, y, z): ({hit_radar_point['x']:.3f}m, {hit_radar_point['y']:.3f}m, {hit_radar_point['z']:.3f}m)")
            print(f"   速度: {hit_radar_point['v_rate']:.2f} m/s")
        else:
            print("未命中散点")
        print("="*40)

def extract_camera_time(img_filename):
    match = re.search(r'frame_(\d+)_(\d+)', img_filename)
    if match:
        return int(match.group(1)) + int(match.group(2)) / 1_000_000.0
    return 0.0

def get_velocity_color(v, v_min=-15.0, v_max=15.0):
    v = np.clip(v, v_min, v_max)
    norm_v = int((v - v_min) / (v_max - v_min) * 255)
    color_img = np.array([[[norm_v]]], dtype=np.uint8)
    color_bgr = cv2.applyColorMap(color_img, cv2.COLORMAP_JET)[0][0]
    return int(color_bgr[0]), int(color_bgr[1]), int(color_bgr[2])

# ==========================================
# 🚀 3. 主函数 
# ==========================================
def main():
    global current_frame_points
    image_folder = r'D:\LEARNING\AY2025-2026T1\MiliwaveRadar\1.26Test\1_26_webcam_frame\Target5'
    excel_file = r'D:\LEARNING\AY2025-2026T1\MiliwaveRadar\1.26Test\ProcessedData\Target5_2.xlsx'

    if not os.path.exists(image_folder) or not os.path.exists(excel_file):
        print("❌ 找不到路径，请检查代码中的路径配置。")
        return

    model = YOLO('yolov8n.pt')
    df = pd.read_excel(excel_file)
    OFFSET = 1769444041
    df['Absolute_Time'] = df['Timestamp_Seconds'] + (df['Timestamp_Nanoseconds'] / 1e9) + OFFSET

    image_paths = glob.glob(os.path.join(image_folder, 'frame_*.jpg'))
    image_paths.sort()

    win_name = 'Radar-Camera Fusion Video'
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1280, 720)
    cv2.setMouseCallback(win_name, on_mouse)

    idx, paused = 0, False

    while idx < len(image_paths):
        img_path = image_paths[idx]
        cam_time = extract_camera_time(os.path.basename(img_path))
        df['Time_Diff'] = np.abs(df['Absolute_Time'] - cam_time)
        min_diff_idx = df['Time_Diff'].idxmin()
        target_sec = df.loc[min_diff_idx, 'Timestamp_Seconds']
        target_nano = df.loc[min_diff_idx, 'Timestamp_Nanoseconds']

        radar_points_df = df[(df['Timestamp_Seconds'] == target_sec) &
                             (df['Timestamp_Nanoseconds'] == target_nano)].copy()

        dt = cam_time - df.loc[min_diff_idx, 'Absolute_Time']
        radar_points_df['x_m'] = radar_points_df['x_m'] + radar_points_df['RangeRate_mS'] * dt
        clean_df = radar_points_df[(radar_points_df['x_m'] > 0.0) & (radar_points_df['x_m'] <= 150.0) & (radar_points_df['RCS_dBm2'] > -12.0)]

        img = cv2.imread(img_path)
        if img is None:
            idx += 1; continue

        results = model(img, verbose=False)[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()
        valid_boxes = [(map(int, box)) for box, cls in zip(boxes, classes) if int(cls) in [2, 5, 7]]

        matched_points, current_frame_points = 0, []

        if len(clean_df) > 0:
            x_r, y_r, z_r = clean_df['x_m'].values, clean_df['y_m'].values, clean_df['z_m'].values
            velocities = clean_df['RangeRate_mS'].values

            pts_3d_radar = np.stack([x_r, y_r, z_r], axis=1)
            pts_3d_shifted = np.dot(R.T, pts_3d_radar.T) + t
            pts_3d_shifted = pts_3d_shifted.T

            # 映射到相机坐标系
            pts_3d_final = np.stack([-pts_3d_shifted[:, 1], pts_3d_shifted[:, 2], pts_3d_shifted[:, 0]], axis=1)
            pts_2d, _ = cv2.projectPoints(pts_3d_final, np.zeros((3,1)), np.zeros((3,1)), K, dist_coeffs)
            pts_2d = pts_2d.reshape(-1, 2)

            for i in range(len(pts_2d)):
                u, v = int(pts_2d[i, 0]), int(pts_2d[i, 1])
                if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
                    current_frame_points.append({'u': u, 'v': v, 'x': x_r[i], 'y': y_r[i], 'z': z_r[i], 'v_rate': velocities[i]})
                    is_hit = any(b[0] <= u <= b[2] and b[1] <= v <= b[3] for b in valid_boxes)
                    if is_hit:
                        matched_points += 1
                        cv2.circle(img, (u, v), 6, get_velocity_color(velocities[i]), -1)
                    else:
                        cv2.circle(img, (u, v), 6, (255, 0, 0), 2)

        info_text = f"{'[PAUSED]' if paused else '[PLAYING]'} Time:{cam_time:.2f} | Hit:{matched_points}/{len(clean_df)}"
        cv2.putText(img, info_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255) if paused else (0, 255, 0), 3)
        cv2.imshow(win_name, img)

        key = cv2.waitKey(0 if paused else 50) & 0xFF
        if key == 27: break
        elif key == 32: paused = not paused
        elif key == ord('d') and paused: idx = min(idx + 1, len(image_paths) - 1)
        elif key == ord('a') and paused: idx = max(idx - 1, 0)
        else:
            if not paused: idx += 1

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()