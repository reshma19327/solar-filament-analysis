import os
import cv2
import numpy as np

MASK_DIR = "data/train/masks"

files = os.listdir(MASK_DIR)

positive_pixels = []
total_pixels = []

for file in files:

    mask = cv2.imread(
        os.path.join(MASK_DIR, file),
        cv2.IMREAD_GRAYSCALE
    )

    # Count filament pixels
    positive = np.sum(mask > 0)

    total = mask.size

    positive_pixels.append(positive)
    total_pixels.append(total)


positive_pixels = np.array(positive_pixels)
total_pixels = np.array(total_pixels)

ratios = positive_pixels / total_pixels

print("Number of masks:", len(files))

print(
    "Average filament percentage:",
    ratios.mean() * 100,
    "%"
)

print(
    "Minimum filament percentage:",
    ratios.min() * 100,
    "%"
)

print(
    "Maximum filament percentage:",
    ratios.max() * 100,
    "%"
)

print(
    "Masks with NO filament pixels:",
    np.sum(positive_pixels == 0)
)