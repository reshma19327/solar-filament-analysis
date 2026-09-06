import cv2
import matplotlib.pyplot as plt
import os

IMAGE_DIR = "data/images"
MASK_DIR = "data/masks"

# Get image filenames
files = os.listdir(IMAGE_DIR)

# Take the first image
filename = files[0]

image = cv2.imread(
    os.path.join(IMAGE_DIR, filename)
)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

mask = cv2.imread(
    os.path.join(MASK_DIR, filename),
    cv2.IMREAD_GRAYSCALE
)

print("Image:", filename)
print("Image shape:", image.shape)
print("Mask shape:", mask.shape)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("256 × 256 Solar Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(mask, cmap="gray")
plt.title("256 × 256 Ground Truth Mask")
plt.axis("off")

plt.tight_layout()
plt.show()