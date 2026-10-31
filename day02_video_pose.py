
from pathlib import Path
import csv
import cv2
from ultralytics import YOLO

# 1. Set up file paths
root = Path(__file__).resolve().parent
video_path = root / "data" / "serve.mp4"
output_dir = root / "outputs"
output_dir.mkdir(exist_ok=True)

csv_path = output_dir / "keypoints.csv"
output_path = output_dir / "pose_output.mp4"

# 2. Load the YOLO pose model
model = YOLO("yolo26n-pose.pt")

# 3. Define the 17 body keypoints
keypoint_names = [
    "nose",
    "left_eye", "right_eye",
    "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle"
]

# 4. Open video
cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    raise FileNotFoundError(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if fps <= 0:
    raise ValueError("Invalid video FPS")

# 5. Create output video
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(
    str(output_path), fourcc, fps, (width, height)
)

if not writer.isOpened():
    raise RuntimeError("Cannot create output video")

frame_id = 0
detected_frames = 0

# 6. Process frames and export keypoints
try:
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow([
            "frame", "time_s", "keypoint_id",
            "keypoint_name", "x_px", "y_px",
            "confidence"
        ])

        while True:
            success, frame = cap.read()

            if not success:
                break

            # Run pose estimation
            result = model.predict(
                frame, verbose=False
            )[0]

            time_s = frame_id / fps

            if (
                result.keypoints is not None
                and len(result.keypoints.xy) > 0
            ):
                # Choose highest-confidence person
                person_idx = int(
                    result.boxes.conf.argmax().item()
                )

                xy = result.keypoints.xy[
                    person_idx
                ].cpu().numpy()

                conf_tensor = result.keypoints.conf
                conf = (
                    conf_tensor[person_idx].cpu().numpy()
                    if conf_tensor is not None
                    else [None] * len(keypoint_names)
                )

                detected_frames += 1

                for k, name in enumerate(keypoint_names):
                    csv_writer.writerow([
                        frame_id,
                        round(time_s, 4),
                        k,
                        name,
                        float(xy[k][0]),
                        float(xy[k][1]),
                        float(conf[k]) if conf[k] is not None else ""
                    ])

            else:
                # Preserve missing frames in the dataset
                for k, name in enumerate(keypoint_names):
                    csv_writer.writerow([
                        frame_id, round(time_s, 4),
                        k, name, "", "", ""
                    ])

            # Draw skeleton on the video
            annotated_frame = result.plot()
            writer.write(annotated_frame)

            frame_id += 1

            if frame_id % 25 == 0:
                print(f"Processed {frame_id} frames")

finally:
    cap.release()
    writer.release()

print("\nProcessing completed!")
print(f"Total processed frames: {frame_id}")
print(f"Frames with detections: {detected_frames}")
print(f"CSV saved to: {csv_path}")
print(f"Video saved to: {output_path}")
