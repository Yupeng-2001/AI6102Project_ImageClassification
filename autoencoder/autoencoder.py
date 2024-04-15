import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision
from constants import *

IMAGE_SIZE = 224
LATENT_DIM = 1024
DECODER_INIT_SIZE = 7
INIT_CHANNEL = 128
RESNET_FEATURE_CHANNEL = 2048
RESNET_FEATURE_MAP_SIZE = 7


class Encoder(nn.Module):
    conv_config = None

    def __init__(self) -> None:
        # use resnet 50 as encoder
        super().__init__()
        # assume 224 * 224 input
        self.encoder = nn.Sequential(
            nn.Conv2d(3, INIT_CHANNEL, 4, stride=2, padding=1),  # 112
            nn.LeakyReLU(),
            nn.Conv2d(INIT_CHANNEL, INIT_CHANNEL * 2, 4, stride=2, padding=1),  # 56
            nn.LeakyReLU(),
            nn.Conv2d(INIT_CHANNEL * 2, INIT_CHANNEL * 4, 4, stride=2, padding=1),  # 28
            nn.LeakyReLU(),
            nn.Conv2d(INIT_CHANNEL * 4, INIT_CHANNEL * 8, 4, stride=2, padding=1),  # 14
            nn.LeakyReLU(),
            nn.Conv2d(INIT_CHANNEL * 8, INIT_CHANNEL * 16, 4, stride=2, padding=1),  # 7
            nn.LeakyReLU(),
        )
        self.conv_head = nn.Sequential(
            nn.Conv2d(INIT_CHANNEL * 16, LATENT_DIM, 7, stride=1, padding=0),
            nn.LeakyReLU(),
        )

        self.feature_head = nn.Linear(LATENT_DIM, LATENT_DIM)  # only output logits

    def forward(self, x: torch.Tensor):
        x = self.encoder(x)
        x = self.conv_head(x)
        x = self.feature_head(x)
        return x


class ResnetEncoder(nn.Module):
    def __init__(self, output_features: bool = True) -> None:
        super().__init__()
        self.output_features = output_features
        # Load the pre-trained ResNet model
        resnet_encoder = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        )
        # Remove the classification layer
        resnet_encoder = nn.Sequential(*(lst := list(resnet_encoder.children())[:-2]))
        out_features = RESNET_FEATURE_CHANNEL
        self.encoder = resnet_encoder
        if output_features:
            # self.conv_head = nn.Conv2d(2048, out_features, 7, stride=1, padding=0)
            self.fn_head = nn.Linear(out_features, LATENT_DIM)
        else:
            # self.conv_head = nn.Identity()
            self.fn_head = nn.Identity()

    def forward(self, x: torch.Tensor):
        x = self.encoder(x)  # [bsz, 2048, 7, 7]
        if self.output_features:
            # x = self.conv_head(x)  # [bsz, 2048, 1, 1]
            x = torch.mean(x, dim=(2, 3))
            # x = x.squeeze(-1).squeeze(-1)  # [bsz, 2048]
            x = self.fn_head(x)
        return x


""" 
class ResnetEncoder(nn.Module):
    def __init__(self, output_features: bool = True) -> None:
        super().__init__()
        self.output_features = output_features
        # Load the pre-trained ResNet model
        resnet_encoder = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        )
        # Remove the classification layer
        resnet_encoder = nn.Sequential(*(lst := list(resnet_encoder.children())[:-2]))
        out_features = 2048
        self.encoder = resnet_encoder
        if output_features:
            self.conv_head = nn.Conv2d(2048, out_features, 7, stride=1, padding=0)
            self.fn_head = nn.Linear(out_features, LATENT_DIM)
        else:
            self.conv_head = nn.Identity()
            self.fn_head = nn.Identity()

    def forward(self, x: torch.Tensor):
        x = self.encoder(x)  # [bsz, 2048, 7, 7]
        if self.output_features:
            x = self.conv_head(x)  # [bsz, 2048, 1, 1]
            x = x.squeeze(-1).squeeze(-1)  # [bsz, 2048]
            x = self.fn_head(x)
        return x
"""


class DownstreamClassifier(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        feed_feature_map: bool = False,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        for param in self.encoder.parameters():
            if freeze_backbone:
                param.requires_grad = False
            else:
                param.requires_grad = True
        self.feed_feature_map = feed_feature_map
        if feed_feature_map:
            # layerwise convolution for each feature
            self.feature_conv_head = nn.Conv2d(
                RESNET_FEATURE_CHANNEL,
                RESNET_FEATURE_CHANNEL,
                kernel_size=RESNET_FEATURE_MAP_SIZE,
                groups=RESNET_FEATURE_CHANNEL,
            )
            feature_size_in = RESNET_FEATURE_CHANNEL
        else:
            self.feature_conv_head = nn.Identity()
            feature_size_in = LATENT_DIM
        self.head = nn.Sequential(
            nn.Linear(feature_size_in, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Linear(256, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor):
        x = self.encoder(x)
        x = self.feature_conv_head(x)
        if self.feed_feature_map:
            x = x.squeeze(-1).squeeze(-1)
        x = self.head(x)
        return x


class Decoder(nn.Module):
    # just deconvolute
    def __init__(
        self,
        feed_feature: bool = True,
    ) -> None:
        super().__init__()
        self.feed_feature = feed_feature
        if feed_feature:
            self.feature_convt = nn.Sequential(
                nn.ConvTranspose2d(
                    LATENT_DIM,
                    INIT_CHANNEL * 16,
                    RESNET_FEATURE_MAP_SIZE,
                    stride=1,
                    padding=0,
                ),  # 7
                nn.LeakyReLU(),
            )
        else:
            self.feature_convt = nn.Identity()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                INIT_CHANNEL * 16, INIT_CHANNEL * 4, 4, stride=2, padding=1  # 14
            ),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(
                INIT_CHANNEL * 4, INIT_CHANNEL * 2, 4, stride=2, padding=1  # 28
            ),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(
                INIT_CHANNEL * 2, INIT_CHANNEL * 1, 4, stride=2, padding=1  # 56
            ),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(
                INIT_CHANNEL * 1, INIT_CHANNEL // 2, 4, stride=2, padding=1  # 128
            ),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(INIT_CHANNEL // 2, 3, 4, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor):
        # expect LATENT_DIM, reshape first
        if self.feed_feature:  # from [bsz, lat] to [bsz, lat, 1, 1]
            x = x.unsqueeze(-1).unsqueeze(-1)
        x = self.feature_convt(x)
        x = self.decoder(x)
        return x


class AutoEncoder(nn.Module):
    def __init__(self, use_feature: bool = True) -> None:
        print(f">>> initializing autoencoder. use feature: {use_feature}")
        super().__init__()
        self.encoder = ResnetEncoder(output_features=use_feature)
        self.decoder = Decoder(feed_feature=use_feature)

    def forward(self, x: torch.Tensor):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
