from ultralytics import YOLO
from pathlib import Path

# Step 1: Load the pretrained YOLO pose estimation model
model = YOLO("yolo26n-pose.pt")

# Step 2: Define the input image path
image_path = "data/test.jpg"

# Step 3: Run human pose estimation
results = model(image_path)

# Step 4: Create the output directory
Path("outputs").mkdir(exist_ok=True)

# Step 5: Save the annotated image
results[0].save(filename="outputs/pose_result.jpg")

# Step 6: Extract human body keypoints
if results[0].keypoints is not None:
    keypoints = results[0].keypoints.xy

    print("Number of detected persons:", keypoints.shape[0])
    print("Number of keypoints per person:", keypoints.shape[1])
    print("Keypoint coordinates:", keypoints)

print("Pose estimation completed successfully!")
