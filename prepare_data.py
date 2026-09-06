import json
import os
import cv2
import numpy as np

JSON_PATH = "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
IMAGE_DIR = "train/train_images"

OUTPUT_IMAGE_DIR = "data/images"
OUTPUT_MASK_DIR = "data/masks"

os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)

with open(JSON_PATH, "r") as f:
    data = json.load(f)

images = data["images"]
annotations = data["annotations"]

images = images[:100]

print("Processing", len(images), "images...")

for index, image_info in enumerate(images):

    image_id = image_info["id"]
    file_name = image_info["file_name"]

    image_path = os.path.join(IMAGE_DIR, file_name)

    image = cv2.imread(image_path)

    if image is None:
        print("Skipping:", file_name)
        continue

    # Resize image
    image = cv2.resize(image, (256, 256))

    original_width = image_info["width"]
    original_height = image_info["height"]

    # Create empty mask
    mask = np.zeros(
        (original_height, original_width),
        dtype=np.uint8
    )

    # Find annotations using image_id
    image_annotations = [
        ann for ann in annotations
        if ann["image_id"] == image_id
    ]

    for ann in image_annotations:

        for polygon in ann["segmentation"]:

            points = np.array(
                polygon,
                dtype=np.float32
            ).reshape(-1, 2)

            points = np.round(points).astype(np.int32)

            cv2.fillPoly(
                mask,
                [points],
                255
            )

    # Resize mask
    mask = cv2.resize(
        mask,
        (256, 256),
        interpolation=cv2.INTER_NEAREST
    )

    # IMPORTANT:
    # Use image_id to make filename unique
    output_name = f"{image_id}.png"

    cv2.imwrite(
        os.path.join(OUTPUT_IMAGE_DIR, output_name),
        image
    )

    cv2.imwrite(
        os.path.join(OUTPUT_MASK_DIR, output_name),
        mask
    )

    print(f"[{index + 1}/100] {output_name}")

print("\nDone!")
print("Images:", len(os.listdir(OUTPUT_IMAGE_DIR)))
print("Masks :", len(os.listdir(OUTPUT_MASK_DIR)))