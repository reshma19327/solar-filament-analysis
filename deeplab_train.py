import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import SolarFilamentDataset
from deeplab_model import DeepLabV3


# ============================================================
# SETTINGS
# ============================================================

TRAIN_IMAGE_DIR = "data/train/images"
TRAIN_MASK_DIR = "data/train/masks"

VAL_IMAGE_DIR = "data/val/images"
VAL_MASK_DIR = "data/val/masks"

MODEL_PATH = "deeplabv3_baseline.pth"

BATCH_SIZE = 4
EPOCHS = 10
LEARNING_RATE = 0.0001

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ============================================================
# DATASET
# ============================================================

train_dataset = SolarFilamentDataset(
    TRAIN_IMAGE_DIR,
    TRAIN_MASK_DIR
)

val_dataset = SolarFilamentDataset(
    VAL_IMAGE_DIR,
    VAL_MASK_DIR
)

print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# MODEL
# ============================================================

model = DeepLabV3(
    pretrained=True
).to(device)


# ============================================================
# LOSS
# ============================================================

class DiceLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, predictions, targets):

        predictions = torch.sigmoid(predictions)

        predictions = predictions.view(-1)
        targets = targets.view(-1)

        intersection = (
            predictions * targets
        ).sum()

        dice = (
            2.0 * intersection + 1.0
        ) / (
            predictions.sum()
            + targets.sum()
            + 1.0
        )

        return 1.0 - dice


dice_loss = DiceLoss()

bce_loss = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor(
        [10.0],
        device=device
    )
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING
# ============================================================

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0

    for images, masks in train_loader:

        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss_bce = bce_loss(
            outputs,
            masks
        )

        loss_dice = dice_loss(
            outputs,
            masks
        )

        loss = loss_bce + loss_dice

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    average_loss = (running_loss /len(train_loader))

    print(f"Epoch [{epoch + 1}/{EPOCHS}] "
          f"Training Loss: {average_loss:.4f}"
    )


# ============================================================
# SAVE MODEL
# ============================================================
torch.save(model.state_dict(),MODEL_PATH)

print()
print("Training completed.")
print("Model saved as:", MODEL_PATH)