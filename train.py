import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import SolarFilamentDataset
from model import UNet


# ==========================================
# 1. Device
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ==========================================
# 2. Dataset
# ==========================================

train_dataset = SolarFilamentDataset(
    "data/train/images",
    "data/train/masks"
)

val_dataset = SolarFilamentDataset(
    "data/val/images",
    "data/val/masks"
)

train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False
)

print("Training samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))


# ==========================================
# 3. Model
# ==========================================

model = UNet().to(device)


# ==========================================
# 4. Dice Loss
# ==========================================

class DiceLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, predictions, targets):

        predictions = torch.sigmoid(predictions)

        predictions = predictions.view(-1)
        targets = targets.view(-1)

        intersection = (predictions * targets).sum()

        dice = (
            2.0 * intersection + 1.0
        ) / (
            predictions.sum()
            + targets.sum()
            + 1.0
        )

        return 1 - dice


# ==========================================
# 5. BCE + Dice Loss
# ==========================================

bce_loss = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor([10.0]).to(device)
)

dice_loss = DiceLoss()


def combined_loss(predictions, targets):

    bce = bce_loss(
        predictions,
        targets
    )

    dice = dice_loss(
        predictions,
        targets
    )

    return bce + dice


# ==========================================
# 6. Optimizer
# ==========================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# ==========================================
# 7. Training
# ==========================================

epochs = 10

for epoch in range(epochs):

    model.train()

    total_loss = 0.0

    for images, masks in train_loader:

        images = images.to(device)
        masks = masks.to(device)

        # Forward
        outputs = model(images)

        # Loss
        loss = combined_loss(
            outputs,
            masks
        )

        # Backpropagation
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    average_loss = (
        total_loss / len(train_loader)
    )

    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Training Loss: {average_loss:.4f}"
    )


# ==========================================
# 8. Save model
# ==========================================

torch.save(
    model.state_dict(),
    "unet_baseline.pth"
)

print("\nTraining completed!")
print("Model saved as: unet_baseline.pth")