
from pathlib import Path
import csv
import cv2

root = Path(__file__).resolve().parent
video_path = root / "data" / "serve.mp4"
csv_path = root / "outputs" / "keypoints.csv"

output_dir = root / "outputs" / "right_arm_check"
output_dir.mkdir(parents=True, exist_ok=True)

# Frames to inspect
selected_frames = [
    184, 232, 233, 234,
    273, 275, 281, 282, 283, 284
]

joint_names = [
    "right_shoulder",
    "right_elbow",
    "right_wrist"
]

# BGR colors
colors = {
    "right_shoulder": (0, 255, 255),
    "right_elbow": (0, 0, 255),
    "right_wrist": (0, 255, 0)
}

# Load keypoint data
data = {}

with open(csv_path, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        frame_id = int(row["frame"])
        data.setdefault(frame_id, {})
        data[frame_id][row["keypoint_name"]] = row

cap = cv2.VideoCapture(str(video_path))
fps = cap.get(cv2.CAP_PROP_FPS)

if not cap.isOpened():
    raise RuntimeError("Cannot open video")

for frame_id in selected_frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    success, image = cap.read()

    if not success:
        continue

    points = {}

    for index, name in enumerate(joint_names):
        row = data.get(frame_id, {}).get(name)

        if row is None or not row["x_px"] or not row["y_px"]:
            continue

        x = int(float(row["x_px"]))
        y = int(float(row["y_px"]))
        conf = float(row["confidence"])

        points[name] = (x, y)

        cv2.circle(image, (x, y), 9, colors[name], -1)

        label = f"{name}: {conf:.2f}"

        cv2.putText(
            image, label,
            (30, 90 + index * 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85, colors[name], 2
        )

    # Draw right upper arm and forearm
    for a, b in [
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist")
    ]:
        if a in points and b in points:
            cv2.line(
                image, points[a], points[b],
                (255, 255, 255), 3
            )

    cv2.putText(
        image,
        f"Frame {frame_id} | {frame_id/fps:.2f}s",
        (30, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1, (255, 255, 255), 2
    )

    output_path = output_dir / f"right_arm_{frame_id}.jpg"
    cv2.imwrite(str(output_path), image)

cap.release()

print("Right arm inspection completed!")
print(f"Saved to: {output_dir}")
