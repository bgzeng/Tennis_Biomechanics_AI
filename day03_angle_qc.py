
from pathlib import Path
import csv
import cv2

root = Path(__file__).resolve().parent
csv_path = root / "outputs" / "joint_angles.csv"
video_path = root / "outputs" / "pose_output.mp4"
review_dir = root / "outputs" / "day03_review"
review_dir.mkdir(parents=True, exist_ok=True)

# 1. Read angle measurements
angles = {}

with open(csv_path, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        frame = int(row["frame"])
        value = row["right_elbow_angle_deg"]
        angles[frame] = float(value) if value else None

# 2. Calculate frame-to-frame angle changes
jumps = []

for frame in sorted(angles):
    current = angles[frame]
    previous = angles.get(frame - 1)

    if current is None or previous is None:
        continue

    change = current - previous

    if abs(change) > 25:
        jumps.append((frame, change))

print("ANGLE JUMP REPORT (>25 deg/frame)")
print(f"Suspicious transitions: {len(jumps)}")

for frame, change in sorted(
    jumps, key=lambda x: abs(x[1]), reverse=True
)[:10]:
    print(f"Frame {frame}: {change:+.2f} degrees")

# 3. Find minimum and maximum angle frames
valid = {
    frame: angle
    for frame, angle in angles.items()
    if angle is not None
}

min_frame = min(valid, key=valid.get)
max_frame = max(valid, key=valid.get)

print(f"\nMinimum at frame {min_frame}")
print(f"Maximum at frame {max_frame}")

# 4. Select frames for visual inspection
cap = cv2.VideoCapture(str(video_path))
if not cap.isOpened():
    raise FileNotFoundError(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    raise ValueError("Invalid video FPS")

review_times = [8.8, 9.0, 9.2, 10.9, 11.1, 11.3]

selected = {
    round(t * fps) for t in review_times
}
selected.update([min_frame, max_frame])

# Include the 5 largest angle jumps
for frame, change in sorted(
    jumps, key=lambda x: abs(x[1]), reverse=True
)[:5]:
    selected.update([frame - 1, frame])

# 5. Export annotated frames
for frame in sorted(selected):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    success, image = cap.read()

    if not success:
        continue

    angle = angles.get(frame)
    angle_text = (
        f"{angle:.1f} deg"
        if angle is not None else "Missing"
    )

    label = (
        f"Frame {frame} | {frame/fps:.2f}s "
        f"| Elbow: {angle_text}"
    )

    cv2.putText(
        image, label, (35, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.3, (0, 255, 255), 3
    )

    output = review_dir / f"frame_{frame:03d}.jpg"
    cv2.imwrite(str(output), image)

cap.release()

print(f"\nReview images saved to: {review_dir}")
