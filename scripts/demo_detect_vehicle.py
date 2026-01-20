import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

# COCO 类别：car=2, bus=5, truck=7
VEHICLE_CLASSES = {2: "car", 5: "bus", 7: "truck"}

def parse_args():
    parser = argparse.ArgumentParser(description="Vehicle detection demo")
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Run without cv2.imshow (for headless environments)",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    model = YOLO("yolov8n.pt")
    video_path = Path(__file__).resolve().parent.parent / "road.mp4"
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频文件: {video_path}")

    

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)[0]

        if results.boxes is not None:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in VEHICLE_CLASSES:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                u = (x1 + x2) // 2
                v = (y1 + y2) // 2

                label = VEHICLE_CLASSES[cls_id]
                conf = float(box.conf[0])
                label_line = f"{label} {conf:.2f}"
                info_line = f"bbox=({x1},{y1},{x2},{y2}) center=({u},{v})"

                # 画框
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # 中心点
                cv2.circle(frame, (u, v), 4, (0, 0, 255), -1)

                # 屏幕显示
                cv2.putText(
                    frame,
                    label_line,
                    (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )
                cv2.putText(
                    frame,
                    info_line,
                    (x1, min(frame.shape[0] - 5, y1 + 14)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

                # 终端输出坐标
                print(f"{label} bbox=({x1},{y1},{x2},{y2}) center=({u},{v})")

        if not args.no_show:
            cv2.imshow("Vehicle Detection (YOLOv8)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if not args.no_show:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
