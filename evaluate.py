import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from model import UNet


# ============================================================
# Settings
# ============================================================

IMAGE_DIR = "data/images"
MASK_DIR = "data/masks"
MODEL_PATH = "unet_baseline.pth"

IMAGE_SIZE = 256
BATCH_SIZE = 4

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ============================================================
# Dataset
# ============================================================

class FilamentDataset(Dataset):

    def __init__(self, image_dir, mask_dir):

        self.image_dir = image_dir
        self.mask_dir = mask_dir

        self.images = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith(".png")
        ])

        print("Number of images:", len(self.images))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        filename = self.images[index]

        image_path = os.path.join(
            self.image_dir,
            filename
        )

        mask_path = os.path.join(
            self.mask_dir,
            filename
        )

        # -----------------------------
        # Load image
        # -----------------------------

        image = Image.open(
            image_path
        ).convert("RGB")

        image = image.resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        )

        image = np.array(
            image,
            dtype=np.float32
        ) / 255.0

        image = torch.tensor(
            image
        ).permute(2, 0, 1)

        # -----------------------------
        # Load mask
        # -----------------------------

        mask = Image.open(
            mask_path
        ).convert("L")

        mask = mask.resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        )

        mask = np.array(
            mask,
            dtype=np.float32
        ) / 255.0

        # Binary mask
        mask = (mask > 0.5).astype(
            np.float32
        )

        mask = torch.tensor(
            mask
        ).unsqueeze(0)

        return image, mask


# ============================================================
# Load dataset
# ============================================================

dataset = FilamentDataset(
    IMAGE_DIR,
    MASK_DIR
)

# Same 80/20 split
generator = torch.Generator().manual_seed(42)

train_size = int(
    0.8 * len(dataset)
)

val_size = len(dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(
    dataset,
    [train_size, val_size],
    generator=generator
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("Validation samples:", len(val_dataset))


# ============================================================
# Load model
# ============================================================

model = UNet().to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()

print("Model loaded successfully")


# ============================================================
# Evaluation
# ============================================================

TP = 0
FP = 0
FN = 0
TN = 0

with torch.no_grad():

    for images, masks in val_loader:

        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)

        probabilities = torch.sigmoid(
            outputs
        )

        predictions = (
            probabilities > 0.5
        ).float()

        # Flatten
        predictions = predictions.view(-1)
        masks = masks.view(-1)

        TP += (
            (predictions == 1) &
            (masks == 1)
        ).sum().item()

        FP += (
            (predictions == 1) &
            (masks == 0)
        ).sum().item()

        FN += (
            (predictions == 0) &
            (masks == 1)
        ).sum().item()

        TN += (
            (predictions == 0) &
            (masks == 0)
        ).sum().item()


# ============================================================
# Metrics
# ============================================================

epsilon = 1e-7

dice = (
    2 * TP
) / (
    2 * TP + FP + FN + epsilon
)

iou = (
    TP
) / (
    TP + FP + FN + epsilon
)

precision = (
    TP
) / (
    TP + FP + epsilon
)

recall = (
    TP
) / (
    TP + FN + epsilon
)

f1 = (
    2 * precision * recall
) / (
    precision + recall + epsilon
)


# ============================================================
# Results
# ============================================================

print("\n================================")
print("BASELINE U-NET RESULTS")
print("================================")

print(f"Dice Score : {dice:.4f}")
print(f"IoU        : {iou:.4f}")
print(f"Precision  : {precision:.4f}")
print(f"Recall     : {recall:.4f}")
print(f"F1 Score   : {f1:.4f}")

print("\nConfusion Matrix")
print("----------------------------")
print("TP:", TP)
print("FP:", FP)
print("FN:", FN)
print("TN:", TN)