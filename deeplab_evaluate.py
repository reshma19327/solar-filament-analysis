import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

from deeplab_model import DeepLabV3


# ============================================================
# CONFIGURATION
# ============================================================
IMAGE_DIR = "data/val/images"
MASK_DIR = "data/val/masks"
MODEL_PATH = "deeplabv3_baseline.pth"

IMAGE_SIZE = 256
BATCH_SIZE = 1
THRESHOLD = 0.5
# ============================================================
# DEVICE
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("========================================")
print("DEEP LAB V3 EVALUATION")
print("========================================")
print("Using device:", device)
# ============================================================
# DATASET
# ============================================================
class FilamentDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir

        self.images = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    def __len__(self):
        return len(self.images)
    def __getitem__(self, index):
        filename = self.images[index]
        image_path = os.path.join(
            self.image_dir,
            filename)
        mask_path = os.path.join(
            self.mask_dir,
            filename)

        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(
                f"Could not read image: {image_path}")

        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

        # Resize image
        image = cv2.resize(image,(IMAGE_SIZE, IMAGE_SIZE))

        # Normalize image
        image = image.astype(np.float32) / 255.0

        # Read mask
        mask = cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not read mask: {mask_path}")

        # Resize mask
        mask = cv2.resize(mask,(IMAGE_SIZE, IMAGE_SIZE),interpolation=cv2.INTER_NEAREST)
        # Convert mask to binary
        mask = (mask > 127).astype(np.float32)
        # Convert to tensors
        image = torch.tensor(image,dtype=torch.float32).permute(2, 0, 1)
        mask = torch.tensor(mask,dtype=torch.float32).unsqueeze(0)
        return image, mask, filename


# ============================================================
# LOAD DATA
# ============================================================
dataset = FilamentDataset(IMAGE_DIR,MASK_DIR)

dataloader = DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=False)
print("Validation samples:", len(dataset))
# ============================================================
# LOAD MODEL
# ============================================================

# IMPORTANT:
# aux_loss=True MUST match the architecture used during training.

model = DeepLabV3(pretrained=False)
checkpoint = torch.load(MODEL_PATH,map_location=device)
model.load_state_dict(checkpoint)
model = model.to(device)
model.eval()

# ============================================================
# METRIC VARIABLES
# ============================================================

TP = 0
TN = 0
FP = 0
FN = 0
total_inference_time = 0.0

# ============================================================
# EVALUATION
# ============================================================

sample_image = None
sample_mask = None
sample_prediction = None

with torch.no_grad():
    for images, masks, filenames in dataloader:
        images = images.to(device)
        masks = masks.to(device)
        # Measure inference time
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = torch.cuda.Event(
            enable_timing=True
        ) if device.type == "cuda" else None

        if device.type == "cuda":
            end_time = torch.cuda.Event(enable_timing=True)
            start_time.record()
        else:
            import time
            cpu_start = time.perf_counter()
        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------
        output = model(images)
        # DeepLabV3 returns a dictionary
        # when using torchvision implementation.
        if isinstance(output, dict):
            output = output["out"]
        probabilities = torch.sigmoid(output)
        predictions = (probabilities > THRESHOLD).float()

        # ----------------------------------------------------
        # INFERENCE TIME
        # ----------------------------------------------------
        if device.type == "cuda":
            end_time.record()
            torch.cuda.synchronize()
            inference_time = (start_time.elapsed_time(end_time)/ 1000.0)

        else:
            import time
            inference_time = (time.perf_counter() - cpu_start)
        total_inference_time += inference_time

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        pred = predictions.cpu().numpy().astype(bool)
        true = masks.cpu().numpy().astype(bool)
        TP += np.logical_and(pred,true).sum()
        TN += np.logical_and(~pred,~true).sum()
        FP += np.logical_and(pred,~true).sum()
        FN += np.logical_and(~pred,true).sum()

        # ----------------------------------------------------
        # SAVE FIRST SAMPLE FOR VISUALIZATION
        # ----------------------------------------------------

        if sample_image is None:
            sample_image = (images[0].cpu().permute(1, 2, 0).numpy())
            sample_mask = (masks[0].cpu().squeeze().numpy())
            sample_prediction = (predictions[0].cpu().squeeze().numpy())


# ============================================================
# CALCULATE METRICS
# ============================================================

epsilon = 1e-8
dice = (2 * TP/(2 * TP + FP + FN + epsilon))

iou = (TP/(TP + FP + FN + epsilon))

precision = (TP/(TP + FP + epsilon))

recall = (TP/(TP + FN + epsilon))

f1 = (2 * precision * recall/(precision + recall + epsilon))

accuracy = ((TP + TN)/(TP + TN + FP + FN + epsilon))

average_inference_time = (total_inference_time/len(dataset))


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("========================================")
print("DEEP LAB V3 RESULTS")
print("========================================")

print(f"Dice Score          : {dice:.4f}")
print(f"IoU                 : {iou:.4f}")
print(f"Precision           : {precision:.4f}")
print(f"Recall              : {recall:.4f}")
print(f"F1 Score            : {f1:.4f}")
print(f"Accuracy            : {accuracy:.4f}")
print(f"Avg Inference Time : "f"{average_inference_time:.4f} seconds/image")
print()
print("Confusion Matrix Values")
print("----------------------------------------")
print("TP:", TP)
print("TN:", TN)
print("FP:", FP)
print("FN:", FN)


# ============================================================
# VISUALIZATION
# ============================================================

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(sample_image)
plt.title("Solar Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(sample_mask, cmap="gray")
plt.title("Ground Truth Mask")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(sample_prediction, cmap="gray")
plt.title("DeepLabV3 Prediction")
plt.axis("off")
plt.tight_layout()
plt.savefig("deeplab_prediction.png",dpi=300,bbox_inches="tight")
plt.show()
print()
print("Prediction visualization saved as:")
print("deeplab_prediction.png")