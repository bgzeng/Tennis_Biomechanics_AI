
from pathlib import Path
import csv
import math
import matplotlib.pyplot as plt

# 1. File paths
root = Path(__file__).resolve().parent
input_path = root / "outputs" / "keypoints.csv"
output_csv = root / "outputs" / "joint_angles.csv"
output_plot = root / "outputs" / "elbow_angle_plot.png"

CONF_THRESHOLD = 0.5

# 2. Calculate the angle between three points
def calculate_angle(shoulder, elbow, wrist):
    # Vectors from elbow to shoulder and wrist
    v1 = (
        shoulder[0] - elbow[0],
        shoulder[1] - elbow[1]
    )
    v2 = (
        wrist[0] - elbow[0],
        wrist[1] - elbow[1]
    )

    dot_product = v1[0] * v2[0] + v1[1] * v2[1]

    norm1 = math.hypot(*v1)
    norm2 = math.hypot(*v2)

    if norm1 == 0 or norm2 == 0:
        return None

    cos_theta = dot_product / (norm1 * norm2)

    # Avoid floating-point errors
    cos_theta = max(-1.0, min(1.0, cos_theta))

    angle_rad = math.acos(cos_theta)
    angle_deg = math.degrees(angle_rad)

    return angle_deg


# 3. Read keypoint data
frames = {}
frame_times = {}

with open(input_path, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for row in reader:
        frame = int(row["frame"])
        name = row["keypoint_name"]

        frames.setdefault(frame, {})
        frames[frame][name] = row
        frame_times[frame] = float(row["time_s"])


# 4. Validate individual keypoints
def get_point(row):
    if row is None:
        return None

    if not all(
        row[field].strip()
        for field in ["x_px", "y_px", "confidence"]
    ):
        return None

    x = float(row["x_px"])
    y = float(row["y_px"])
    conf = float(row["confidence"])

    if not all(math.isfinite(v) for v in (x, y, conf)):
        return None

    if conf < CONF_THRESHOLD or x <= 0 or y <= 0:
        return None

    return (x, y)


# 5. Calculate right elbow angle frame by frame
results = []
valid_angles = []

for frame in sorted(frames):
    data = frames[frame]

    shoulder = get_point(data.get("right_shoulder"))
    elbow = get_point(data.get("right_elbow"))
    wrist = get_point(data.get("right_wrist"))

    angle = None

    if shoulder and elbow and wrist:
        angle = calculate_angle(shoulder, elbow, wrist)

    time_s = frame_times[frame]

    results.append({
        "frame": frame,
        "time_s": time_s,
        "right_elbow_angle_deg": (
            round(angle, 2) if angle is not None else ""
        )
    })

    if angle is not None:
        valid_angles.append(angle)


# 6. Save angles to CSV
with open(output_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "frame", "time_s", "right_elbow_angle_deg"
        ]
    )
    writer.writeheader()
    writer.writerows(results)


# 7. Plot the angle time series
times = [r["time_s"] for r in results]
angles = [
    r["right_elbow_angle_deg"]
    if r["right_elbow_angle_deg"] != ""
    else math.nan
    for r in results
]

plt.figure(figsize=(12, 5))
plt.plot(
    times, angles,
    linewidth=1.5,
    label="Right Elbow Angle"
)

plt.title("Right Elbow Angle During Tennis Serve")
plt.xlabel("Time (seconds)")
plt.ylabel("2D Elbow Angle (degrees)")
plt.ylim(0, 180)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(output_plot, dpi=300)
plt.close()


# 8. Print summary
total = len(results)
valid = len(valid_angles)

print("\nDAY 3 - RIGHT ELBOW ANGLE ANALYSIS")
print(f"Total frames: {total}")
print(f"Valid angle frames: {valid}")
print(f"Angle coverage: {valid / total:.1%}")

if valid_angles:
    print(f"Minimum angle: {min(valid_angles):.2f} degrees")
    print(f"Maximum angle: {max(valid_angles):.2f} degrees")

print(f"\nCSV saved: {output_csv}")
print(f"Plot saved: {output_plot}")
