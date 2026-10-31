
from pathlib import Path
from collections import defaultdict
import csv
import math

# 1. File paths
root = Path(__file__).resolve().parent
csv_path = root / "outputs" / "keypoints.csv"
report_path = root / "outputs" / "qc_summary.csv"

# 2. Quality thresholds
CONF_THRESHOLD = 0.5
JUMP_THRESHOLD = 150  # pixels per frame
WIDTH = 1920
HEIGHT = 1080

# 3. Load CSV
with open(csv_path, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

frames = {int(row["frame"]) for row in rows}
total_frames = len(frames)

if total_frames == 0:
    raise ValueError("CSV contains no frames")

# 4. Organize valid keypoints
valid_points = defaultdict(list)
all_names = set()

for row in rows:
    name = row["keypoint_name"]
    frame = int(row["frame"])
    all_names.add(name)

    if not all(
        row[field].strip()
        for field in ["x_px", "y_px", "confidence"]
    ):
        continue

    x = float(row["x_px"])
    y = float(row["y_px"])
    conf = float(row["confidence"])

    valid = (
        conf >= CONF_THRESHOLD
        and 0 < x < WIDTH
        and 0 < y < HEIGHT
    )

    if valid:
        valid_points[name].append((frame, x, y, conf))

# 5. Calculate quality metrics
summary = []

for name in sorted(all_names):
    points = sorted(valid_points[name])

    detection_rate = len(points) / total_frames
    avg_conf = (
        sum(p[3] for p in points) / len(points)
        if points else 0
    )

    jumps = 0

    for previous, current in zip(points, points[1:]):
        f1, x1, y1, _ = previous
        f2, x2, y2, _ = current

        if f2 != f1 + 1:
            continue

        distance = math.hypot(x2 - x1, y2 - y1)

        if distance > JUMP_THRESHOLD:
            jumps += 1

    summary.append({
        "keypoint": name,
        "detection_rate": round(detection_rate, 3),
        "avg_confidence": round(avg_conf, 3),
        "large_jumps": jumps
    })

# 6. Save quality report
with open(report_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "keypoint", "detection_rate",
        "avg_confidence", "large_jumps"
    ])
    writer.writeheader()
    writer.writerows(summary)

# 7. Print selected joints
important_joints = [
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee"
]

print(f"Total frames: {total_frames}")
print(f"Total CSV rows: {len(rows)}")
print("\nKEYPOINT QUALITY REPORT")

for item in summary:
    if item["keypoint"] in important_joints:
        print(
            f"{item['keypoint']:16s} "
            f"Detection: {item['detection_rate']:.1%} "
            f"Confidence: {item['avg_confidence']:.3f} "
            f"Jumps: {item['large_jumps']}"
        )

print(f"\nReport saved: {report_path}")
