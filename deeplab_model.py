import torch
import torch.nn as nn
from torchvision.models.segmentation import (
    deeplabv3_resnet50,
    DeepLabV3_ResNet50_Weights
)


class DeepLabV3(nn.Module):

    def __init__(self, pretrained=True):

        super().__init__()

        # Load DeepLabV3 with pretrained ResNet-50 backbone
        if pretrained:

            weights = DeepLabV3_ResNet50_Weights.DEFAULT

            self.model = deeplabv3_resnet50(
                weights=weights,
                aux_loss=True
            )

        else:

            self.model = deeplabv3_resnet50(
                weights=None,
                aux_loss=True
            )

        # Change main classifier to 1 output channel
        classifier = self.model.classifier

        classifier[4] = nn.Conv2d(
            in_channels=256,
            out_channels=1,
            kernel_size=1
        )

        self.model.classifier = classifier

        # Change auxiliary classifier to 1 output channel
        if self.model.aux_classifier is not None:

            aux_classifier = self.model.aux_classifier

            aux_classifier[4] = nn.Conv2d(
                in_channels=256,
                out_channels=1,
                kernel_size=1
            )

            self.model.aux_classifier = aux_classifier


    def forward(self, x):

        output = self.model(x)

        return output["out"]


# ============================================================
# TEST MODEL
# ============================================================

if __name__ == "__main__":

    model = DeepLabV3(pretrained=False)

    x = torch.randn(
        2,
        3,
        256,
        256
    )

    output = model(x)

    print("Input shape :", x.shape)
    print("Output shape:", output.shape)