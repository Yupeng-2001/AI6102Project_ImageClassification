import torchvision.models as models
import torch.nn as nn

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
        return x