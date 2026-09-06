import torch
import cv2
import matplotlib.pyplot as plt
import os

from model import UNet


# ==========================================
# 1. Device
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ==========================================
# 2. Load trained model
# ==========================================

model = UNet().to(device)

model.load_state_dict(
    torch.load(
        "unet_baseline.pth",
        map_location=device
    )
)

model.eval()

print("Model loaded successfully")


# ==========================================
# 3. Select validation image
# ==========================================

image_dir = "data/val/images"
mask_dir = "data/val/masks"

files = sorted(os.listdir(image_dir))

filename = files[0]

print("Testing image:", filename)


# ==========================================
# 4. Load image
# ==========================================

image = cv2.imread(
    os.path.join(image_dir, filename)
)

image_rgb = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)

# Normalize
input_image = image_rgb / 255.0


# ==========================================
# 5. Convert image to tensor
# ==========================================

input_tensor = torch.tensor(
    input_image,
    dtype=torch.float32
).permute(2, 0, 1)

input_tensor = input_tensor.unsqueeze(0).to(device)


# ==========================================
# 6. Model prediction
# ==========================================

with torch.no_grad():

    output = model(input_tensor)

    # Convert logits to probability
    probability = torch.sigmoid(output)


# ==========================================
# 7. Print probability statistics
# ==========================================

print("\nProbability statistics:")

print(
    "Minimum probability:",
    probability.min().item()
)

print(
    "Maximum probability:",
    probability.max().item()
)

print(
    "Mean probability:",
    probability.mean().item()
)


# ==========================================
# 8. Convert probability map to NumPy
# ==========================================

probability_map = (
    probability
    .squeeze()
    .cpu()
    .numpy()
)


# ==========================================
# 9. Load ground truth
# ==========================================

mask = cv2.imread(
    os.path.join(
        mask_dir,
        filename
    ),
    cv2.IMREAD_GRAYSCALE
)


# ==========================================
# 10. Create visualization
# ==========================================

plt.figure(figsize=(15, 5))


# Original image
plt.subplot(1, 3, 1)

plt.imshow(image_rgb)

plt.title("Solar Image")

plt.axis("off")


# Ground truth
plt.subplot(1, 3, 2)

plt.imshow(
    mask,
    cmap="gray"
)

plt.title("Ground Truth")

plt.axis("off")


# Raw probability
plt.subplot(1, 3, 3)

plt.imshow(
    probability_map,
    cmap="gray",
    vmin=0,
    vmax=1
)

plt.title("Prediction Probability")

plt.axis("off")


plt.tight_layout()


# ==========================================
# 11. Save result
# ==========================================

plt.savefig(
    "prediction_probability.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print(
    "\nPrediction saved as: "
    "prediction_probability.png"
)