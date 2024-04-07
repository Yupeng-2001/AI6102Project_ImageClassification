import argparse

from data import get_dataloader, default_transform, canny_transform
from model import ResNetClassifier, ViTClassifier, SwinTransformerClassifier
from utils import reset_seeds, save_model
from train import train_model

import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Subset, random_split
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler


def parse_args():
    parser = argparse.ArgumentParser(description="Unzip a file.")

    # Add arguments
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument(
        "--beta1",
        type=float,
        default=0.9,
        help="decay rate for the first moment estimate for Adam",
    )
    parser.add_argument(
        "--beta2", type=float, default=0.999, help="exponential decay rate for Adam"
    )
    parser.add_argument("--momentum", type=float, default=0.1, help="momentum for SGD")
    parser.add_argument(
        "--optimizer_type", type=str, choices=["SGD", "Adam"], default="Adam"
    )
    parser.add_argument(
        "--scheduler_type",
        type=str,
        choices=["cos_annealing", "none"],
        default="cos_annealing",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--transform",
        type=str,
        choices=["default_transform", "canny_transform"],
        default="default_transform",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="resnet101",
        help="model type, specify exact model type here",
    )
    parser.add_argument(
        "--freeze_backbone",
        type=str,
        choices=["y", "n"],
        default="n",
        help="whether to freeze the backbone",
    )
    parser.add_argument("--train_data_path", type=str, help="Path to load data")
    parser.add_argument(
        "--model_save_path", type=str, help="Path to save the model's weight"
    )

    parser.add_argument("--pretrained_weight", type=str, default=None)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # Parse arguments
    args = parse_args()

    batch_size = args.batch_size
    epochs = args.epochs

    optimizer_type = args.optimizer_type
    lr = args.lr
    beta1 = args.beta1
    beta2 = args.beta2
    momentum = args.momentum
    scheduler_type = args.scheduler_type

    model_type = args.model
    if args.freeze_backbone == "y":
        freeze_backbone = True
    else:
        freeze_backbone = False
    model_save_path = args.model_save_path
    train_data_path = args.train_data_path

    seed = args.seed
    num_classes = 121

    reset_seeds(seed)

    """## data loading ##"""
    transform = default_transform
    if args.transform == "canny_transform":
        transform = canny_transform

    dataloader = get_dataloader(train_data_path, batch_size, True, transform)
    validation_size = int(0.1 * len(dataloader.dataset))
    train_dataset, val_dataset = random_split(
        dataloader.dataset, [len(dataloader.dataset) - validation_size, validation_size]
    )

    # Create new DataLoaders for the training and validation datasets
    train_loader = DataLoader(
        train_dataset,
        batch_size=dataloader.batch_size,
        shuffle=True,
        num_workers=dataloader.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=dataloader.batch_size,
        shuffle=False,
        num_workers=dataloader.num_workers,
    )

    """##models, loss and optimizer set up"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = None
    if "resnet" in model_type:
        model = ResNetClassifier(
            num_classes=num_classes,
            freeze_backbone=freeze_backbone,
            model_type=model_type,
            pretrained_weight=args.pretrained_weight,
        )
    elif model_type == "ViT":
        model = ViTClassifier(num_classes=num_classes, freeze_backbone=freeze_backbone)
    elif model_type == "SwinTransformer":
        model = SwinTransformerClassifier(num_classes=num_classes, freeze_backbone=freeze_backbone)
    else:
        model = ViTClassifier(num_classes=num_classes, freeze_backbone=freeze_backbone)
    print(
        f"trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )
    print(f"type of model: {type(model)} from model type {model_type}")

    criterion = nn.CrossEntropyLoss()

    optimizer = None
    if optimizer_type == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2))
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    scheduler = None
    if scheduler_type == "cos_annealing":
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    """##training##"""
    trianing_result, (best_valid_loss, best_model_params) = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        model_save_path,
        model_type,
        scheduler,
        num_epochs=epochs,
        device=device,
    )
    model.load_state_dict(best_model_params)
    save_model(
        model_save_path,
        model_type,
        trianing_result,
        best_valid_loss,
        model,
        dataloader.dataset.classes,
    )