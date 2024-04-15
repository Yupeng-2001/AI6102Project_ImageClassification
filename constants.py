from torchvision import transforms
import torch
from PIL import Image

NUM_CLASSES = 121

# mean and variance for aspect ratio preserved resize only dataset
mean_tr = torch.tensor([0.9414, 0.9414, 0.9414])
std_tr = torch.tensor([0.1496, 0.1496, 0.1496])
DEFAULT_SIZE = 224
AE_SIZE = 96


def pad_to_square(image):
    width, height = image.size
    shorter_side = max(width, height)
    padded_image = Image.new("RGB", (shorter_side, shorter_side), color="white")
    left_padding = (shorter_side - width) // 2
    top_padding = (shorter_side - height) // 2
    padded_image.paste(image, (left_padding, top_padding))
    return padded_image


image_size_width_mean = 73.0799309815951
image_size_height_mean = 66.66404141104294
image_size_width_var = 2390.739155516708
image_size_height_var = 1827.9280750780397

default_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(DEFAULT_SIZE, scale=(0.3, 1)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Pad(50, 255, padding_mode="constant"),
        transforms.RandomRotation(degrees=45),
        transforms.CenterCrop(DEFAULT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_tr, std=std_tr),
    ]
)


def default_pad(x):
    x = pad_to_square(x)
    x = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                DEFAULT_SIZE, scale=(0.3, 1), ratio=(0.9, 1.1)
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Pad(50, 255, padding_mode="constant"),
            transforms.RandomRotation(degrees=45),
            transforms.CenterCrop(DEFAULT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_tr, std=std_tr),
        ]
    )(x)
    return x


default_no_normalize = transforms.Compose(
    [
        transforms.RandomResizedCrop(DEFAULT_SIZE, scale=(0.3, 1)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Pad(50, 255, padding_mode="constant"),
        transforms.RandomRotation(degrees=45),
        transforms.CenterCrop(DEFAULT_SIZE),
        transforms.ToTensor(),
        # transforms.Normalize(mean=mean_tr, std=std_tr),
    ]
)


def ae_transform(x):
    x = pad_to_square(x)
    x = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                DEFAULT_SIZE, scale=(0.95, 1), ratio=(0.95, 1.05)
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Pad(50, 255, padding_mode="constant"),
            transforms.RandomRotation(degrees=45),
            transforms.CenterCrop(DEFAULT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_tr, std=std_tr),
        ]
    )(x)
    return x


# ae_transform = transforms.Compose(
#     [
#         transforms.RandomResizedCrop(DEFAULT_SIZE, scale=(0.8, 1), ratio=(0.95, 1.05)),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomVerticalFlip(),
#         transforms.Pad(50, 255, padding_mode="constant"),
#         transforms.RandomRotation(degrees=45),
#         transforms.CenterCrop(DEFAULT_SIZE),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=mean_tr, std=std_tr),
#     ]
# )


def unnormalize(ts: torch.Tensor, mean, std):
    for i in range(3):  # Iterate over each channel
        ts[:, i, :, :] = ts[:, i, :, :] * std[i] + mean[i]
    return ts


inference_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(DEFAULT_SIZE, scale=(1, 1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_tr, std=std_tr),
    ]
)


def inference_pad(x):
    x = pad_to_square(x)
    x = transforms.Compose(
        [
            transforms.Resize(DEFAULT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_tr, std=std_tr),
        ]
    )(x)
    return x


canny_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(224, scale=(0.3, 1)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=45),
        transforms.ToTensor(),
    ]
)

small_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(96, scale=(0.3, 1)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=45),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_tr, std=std_tr),
    ]
)
