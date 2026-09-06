import os
import cv2
import torch
from torch.utils.data import Dataset


class SolarFilamentDataset(Dataset):

    def __init__(self, image_dir, mask_dir):

        self.image_dir = image_dir
        self.mask_dir = mask_dir

        self.images = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        filename = self.images[index]

        # Load image
        image_path = os.path.join(
            self.image_dir,
            filename
        )

        image = cv2.imread(image_path)

        # Convert BGR → RGB
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        # Normalize image
        image = image / 255.0

        # Load mask
        mask_path = os.path.join(
            self.mask_dir,
            filename
        )

        mask = cv2.imread(
            mask_path,
            cv2.IMREAD_GRAYSCALE
        )

        # Convert mask to 0/1
        mask = mask / 255.0

        # Convert to PyTorch format
        image = torch.tensor(
            image,
            dtype=torch.float32
        ).permute(2, 0, 1)

        mask = torch.tensor(
            mask,
            dtype=torch.float32
        ).unsqueeze(0)

        return image, mask


# Test dataset
dataset = SolarFilamentDataset(
    "data/images",
    "data/masks"
)

print("Number of images:", len(dataset))

image, mask = dataset[0]

print("Image tensor shape:", image.shape)
print("Mask tensor shape:", mask.shape)