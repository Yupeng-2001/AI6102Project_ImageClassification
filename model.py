import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vision_transformer

class ResNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ResNetClassifier, self).__init__()

        # Load pre-trained ResNet50 model
        self.resnet = models.resnet50(pretrained=True)

        # Replace the final fully connected layer with a new one
        num_ftrs = self.resnet.fc.out_features
        self.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        x = self.fc(x)
        return x
    


class ViTClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ViTClassifier, self).__init__()
        self.backbone = vision_transformer.ViT('B_16_imagenet1k', pretrained=True)
        num_ftrs = self.backbone.head.in_features
        self.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x
    

