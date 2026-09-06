import cv2
import matplotlib.pyplot as plt

# Load original image
image = cv2.imread("train/train_images/20140609195854Bh.jpeg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Load mask
mask = cv2.imread("test_mask.png", cv2.IMREAD_GRAYSCALE)

# Display
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Solar Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(mask, cmap="gray")
plt.title("Ground Truth Filament Mask")
plt.axis("off")

plt.tight_layout()
plt.show()