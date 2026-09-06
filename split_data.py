import os
import shutil
import random

IMAGE_DIR = "data/images"
MASK_DIR = "data/masks"

TRAIN_IMAGE_DIR = "data/train/images"
TRAIN_MASK_DIR = "data/train/masks"

VAL_IMAGE_DIR = "data/val/images"
VAL_MASK_DIR = "data/val/masks"

# Create folders
for folder in [
    TRAIN_IMAGE_DIR,
    TRAIN_MASK_DIR,
    VAL_IMAGE_DIR,
    VAL_MASK_DIR
]:
    os.makedirs(folder, exist_ok=True)

# Get images
files = os.listdir(IMAGE_DIR)

# Make results reproducible
random.seed(42)
random.shuffle(files)

# 80% train, 20% validation
split = int(0.8 * len(files))

train_files = files[:split]
val_files = files[split:]

print("Training images:", len(train_files))
print("Validation images:", len(val_files))

# Copy training files
for file in train_files:

    shutil.copy(
        os.path.join(IMAGE_DIR, file),
        os.path.join(TRAIN_IMAGE_DIR, file)
    )

    shutil.copy(
        os.path.join(MASK_DIR, file),
        os.path.join(TRAIN_MASK_DIR, file)
    )

# Copy validation files
for file in val_files:

    shutil.copy(
        os.path.join(IMAGE_DIR, file),
        os.path.join(VAL_IMAGE_DIR, file)
    )

    shutil.copy(
        os.path.join(MASK_DIR, file),
        os.path.join(VAL_MASK_DIR, file)
    )

print("Split completed!")