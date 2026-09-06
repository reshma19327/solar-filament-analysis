import json
import os
import cv2
import numpy as np

# Paths
JSON_PATH = "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
IMAGE_DIR = "train/train_images"

# Load JSON
with open(JSON_PATH, "r") as f:
    data = json.load(f)

# Get first image
image_info = data["images"][0]

image_id = image_info["id"]
file_name = image_info["file_name"]

print("Image ID:", image_id)
print("File name:", file_name)
print("Size:", image_info["width"], "x", image_info["height"])

# Load image
image_path = os.path.join(IMAGE_DIR, file_name)
image = cv2.imread(image_path)

if image is None:
    print("ERROR: Image not found:", image_path)
    exit()

height, width = image.shape[:2]

# Empty mask
mask = np.zeros((height, width), dtype=np.uint8)

# Find annotations belonging to this image
annotations = [
    ann for ann in data["annotations"]
    if ann["image_id"] == image_id
]

print("Number of annotations:", len(annotations))

# Draw each polygon
for ann in annotations:

    segmentation = ann["segmentation"]

    for polygon in segmentation:

        points = np.array(polygon, dtype=np.float32)
        points = points.reshape((-1, 2))

        points = np.round(points).astype(np.int32)

        cv2.fillPoly(mask, [points], 255)

# Save mask
cv2.imwrite("test_mask.png", mask)

print("Mask saved as test_mask.png")