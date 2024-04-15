import torchvision.models as models
from torchvision.models import (
    ResNet50_Weights,
    vision_transformer,
    ViT_B_32_Weights,
    ResNet101_Weights,
    ResNet152_Weights,
    swin_b,
    Swin_B_Weights,
)
import torch.nn as nn
import torch.nn.functional as F


def init_backbone_kaiming(module):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)


class ResNetClassifier(nn.Module):
    str_to_weight_mapping = {
        "resnet50": ResNet50_Weights.IMAGENET1K_V2,
        "resnet101": ResNet101_Weights.IMAGENET1K_V2,
        "resnet152": ResNet152_Weights.IMAGENET1K_V2,
    }
    str_to_model_mapping = {
        "resnet50": models.resnet50,
        "resnet101": models.resnet101,
        "resnet152": models.resnet152,
    }

    def __init__(
        self,
        num_classes,
        freeze_backbone=False,
        model_type: str = "resnet101",
        pretrained_weight=None,
    ):
        super(ResNetClassifier, self).__init__()
        assert not (
            freeze_backbone and pretrained_weight is None
        ), f"if want to use pretrained must not freeze backbone"
        # Load pre-trained ResNet50 model
        if pretrained_weight:
            if isinstance(pretrained_weight, str):
                pretrained_weight = self.str_to_weight_mapping[pretrained_weight]
            # ResNet50_Weights.IMAGENET1K_V2
            self.resnet = self.str_to_model_mapping[model_type](
                weights=pretrained_weight
            )
        else:
            self.resnet = self.str_to_model_mapping[model_type](weights=None)
        self.resnet.fc == nn.Identity()

        if freeze_backbone:
            for param in self.resnet.parameters():
                param.requires_grad = False
        else:
            if pretrained_weight is None:
                self.resnet.apply(init_backbone_kaiming)

        # Replace the final fully connected layer with a new one
        num_ftrs = self.resnet.fc.out_features

        embedding1 = 1024
        embedding2 = 256
        self.fc = nn.Sequential(
            nn.Linear(num_ftrs, embedding1),
            nn.BatchNorm1d(embedding1),
            nn.GELU(),
            nn.Linear(embedding1, embedding2),
            nn.BatchNorm1d(embedding2),
            nn.GELU(),
            nn.Linear(embedding2, num_classes),
        )

    def forward(self, x):
        x = self.resnet(x)
        x = self.fc(x)
        return x


class ViTClassifier(nn.Module):
    def __init__(self, num_classes, freeze_backbone=False):
        super(ViTClassifier, self).__init__()
        self.vit = vision_transformer.vit_b_32(
            weights=ViT_B_32_Weights.IMAGENET1K_V1
        )  # vision_transformer.ViT('B_16_imagenet1k', pretrained=True)
        num_ftrs = self.vit.heads.head.out_features

        self.vit.heads.head == nn.Identity()

        if freeze_backbone:
            for param in self.vit.parameters():
                param.requires_grad = False

        embedding1 = 512
        embedding2 = 256
        self.fc = nn.Sequential(
            nn.Linear(num_ftrs, embedding1),
            nn.BatchNorm1d(embedding1),
            nn.GELU(),
            nn.Linear(embedding1, embedding2),
            nn.BatchNorm1d(embedding2),
            nn.GELU(),
            nn.Linear(embedding2, num_classes),
        )

    def forward(self, x):
        x = self.vit(x)
        x = self.fc(x)
        return x


class SwinTransformerClassifier(nn.Module):
    def __init__(self, num_classes, freeze_backbone=False):
        super(SwinTransformerClassifier, self).__init__()
        self.backbone = swin_b(weights=Swin_B_Weights.IMAGENET1K_V1)
        num_ftrs = self.backbone.head.in_features

        self.backbone.head = nn.Identity()

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        embedding1 = 512
        embedding2 = 256
        self.fc = nn.Sequential(
            nn.Linear(num_ftrs, embedding1),
            nn.BatchNorm1d(embedding1),
            nn.GELU(),
            nn.Linear(embedding1, embedding2),
            nn.BatchNorm1d(embedding2),
            nn.GELU(),
            nn.Linear(embedding2, num_classes),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x
