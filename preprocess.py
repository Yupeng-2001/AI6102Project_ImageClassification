from PIL import Image
import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from typing import Callable
from PIL import Image


def resize_keep_aspect_ratio(
    img: Image.Image, target_size: tuple[int, int] = (224, 224)
) -> Image.Image:
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
    return final_img


def apply_canny(
    image: np.ndarray | Image.Image,
    gaussian_size: tuple = (1, 1),
    canny_low: int = 60,
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


def preprocess_canny(
    img_root: str,
    save_root: str,
    target_size: tuple[int, int] = (224, 224),
    gaussian_size: tuple[int, int] = [1, 1],
    canny_low: int = 50,
    canny_high: int = 140,
):
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    for root, dirs, files in os.walk(img_root):
        for dir in dirs:
            cur_relative_path = os.path.relpath(os.path.join(root, dir), img_root)
            os.makedirs(os.path.join(save_root, cur_relative_path), exist_ok=True)
        for file in files:
            if file.endswith(".jpg"):
                cur_img_path = os.path.join(root, file)
                cur_relative_path = os.path.relpath(cur_img_path, img_root)
                with Image.open(cur_img_path) as img:
                    img_res = resize_keep_aspect_ratio(img, target_size)
                    img_can = apply_canny(
                        img_res, gaussian_size, canny_low, canny_high, out_type="image"
                    )
                    img_can.save(os.path.join(save_root, cur_relative_path))
