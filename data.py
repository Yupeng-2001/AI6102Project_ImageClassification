from PIL import Image
import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from typing import Callable

from torchvision import transforms
from constants import *


def resize_keep_aspect_ratio(
    input_path: str, output_path: str, target_size: tuple[int, int] = (224, 224)
):
    with Image.open(input_path) as img:
        width_ratio = target_size[0] / img.width
        height_ratio = target_size[1] / img.height
        scale_factor = min(width_ratio, height_ratio)
        new_size = (int(img.width * scale_factor), int(img.height * scale_factor))
        resized_img = img.resize(new_size)
        final_img = Image.new("RGB", target_size, (255, 255, 255))
        resized_img.thumbnail(target_size)
        position = (
            (target_size[0] - resized_img.size[0]) // 2,
            (target_size[1] - resized_img.size[1]) // 2,
        )
        final_img.paste(resized_img, position)
        final_img.save(output_path)


def apply_canny(
    image: np.ndarray | Image.Image,
    gaussian_size: tuple = (1, 1),
    canny_low: int = 75,
    canny_high: int = 150,
    out_type: str = "image",
) -> np.ndarray | Image.Image:
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    # [h, w, c]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # [h w]
    blurred = cv2.GaussianBlur(gray, gaussian_size, 0)
    edges = cv2.Canny(blurred, canny_low, canny_high)
    if out_type == "image":
        edges = Image.fromarray(edges)
    elif out_type == "numpy":
        pass
    else:
        pass
    return edges


class AEImageDataset(Dataset):
    def __init__(self, root_dir, transform):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = os.listdir(root_dir)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name).convert(
            "RGB"
        )  # Ensure all images are converted to RGB format
        image = self.transform(image)
        return image, image.clone()  # for ae training image is label


def get_dataset(
    root: str,
    mode: str = "default",
    transform: Callable | None = default_transform,
    target_transform: Callable | None = None,
):
    if "ae" in mode:
        ret = AEImageDataset(root, transform)
    else:
        ret = ImageFolder(root, transform, target_transform)
    return ret


def get_dataloader(
    root: str,
    batch_size: int,
    mode: str = "default",
    shuffle: bool = True,
    transform: Callable | None = default_transform,
    target_transform: Callable | None = None,
    *args,
    **kwargs
):
    dataset = get_dataset(root, mode, transform, target_transform)
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, *args, **kwargs)
    return dl
