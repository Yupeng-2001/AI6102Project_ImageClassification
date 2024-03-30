import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vision_transformer

class ResNetClassifier(nn.Module):
    def __init__(self, num_classes, freeze_backbone = False):
        super(ResNetClassifier, self).__init__()

        # Load pre-trained ResNet50 model
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc == nn.Identity()

        if freeze_backbone:
            for param in self.resnet.parameters():
                param.requires_grad = False

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
    def __init__(self, num_classes, freeze_backbone = False):
        super(ViTClassifier, self).__init__()
        self.vit = vision_transformer.vit_b_16(weights='IMAGENET1K_V1')#vision_transformer.ViT('B_16_imagenet1k', pretrained=True)
        num_ftrs = self.vit.heads.head.out_features

        self.vit.heads.head == nn.Identity()

        if freeze_backbone:
            for param in self.vit.parameters():
                param.requires_grad = False

        embedding1 = 512
        embedding2 = 256
        self.fc = nn.Sequential(
            nn.Linear(num_ftrs, embedding1),
            nn.BatchNorm1d(num_ftrs),
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

