import argparse
import json

from data import get_dataloader
from model import ResNetClassifier, ViTClassifier, SwinTransformerClassifier
from utils import reset_seeds, save_model
from train import train_model
from constants import *
from autoencoder.autoencoder import *

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
    parser.add_argument("--criterion", type=str, default="CrossEntropy")
    parser.add_argument("--dataset_mode", type=str, default="default")
    parser.add_argument("--use_feature", action="store_true")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # Parse arguments
    args = parse_args()

    # Convert namespace object to dictionary
    namespace_dict = vars(args)

    # Specify the file path where you want to save the JSON file
    file_path = "running_args.json"

    # Save dictionary as a JSON file
    with open(file_path, "w") as json_file:
        json.dump(namespace_dict, json_file, indent=4)
    batch_size = args.batch_size
    epochs = args.epochs
    print(f" current args: \n {args}")

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
    if args.transform == "canny_transform":
        transform = canny_transform
    elif args.transform == "small_transform":
        transform = small_transform
    elif args.transform == "ae_transform":
        transform = ae_transform
    elif args.transform == "default_pad":
        transform = default_pad
    else:
        print(f"!!! WARNING: use default transform. maybe not desired")
        transform = default_transform

    dataloader = get_dataloader(
        train_data_path,
        batch_size,
        mode=args.dataset_mode,
        shuffle=True,
        transform=transform,
    )
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
    elif model_type == "SwinTransformer" or "swin" in model_type:
        model = SwinTransformerClassifier(
            num_classes=num_classes, freeze_backbone=freeze_backbone
        )
    elif "ae" in model_type or "AutoEncoder" in model_type:
        model = AutoEncoder(use_feature=args.use_feature)
    elif "enc" in model_type or ("downsteam" in model_type and "ae" in model_type):
        loaded_model = AutoEncoder(args.use_feature)
        loaded_ckpt = torch.load(
            f"/mnt/slurm_home/pzzhao/acad_projects/AI6102_proj/AI6102Project_ImageClassification/model_ckpt/ae_no_feature/model_ep7.pt"
        )
        loaded_model.load_state_dict(loaded_ckpt)
        model = DownstreamClassifier(
            loaded_model.encoder, not args.use_feature, args.freeze_backbone
        )
    else:
        raise NotImplementedError
    print(f">>>")
    print(f"total parameters: {sum(p.numel() for p in model.parameters())}")
    print(
        f"trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )
    print(f"type of model: {type(model)} from model type {model_type}")
    print(f">>>")

    if "CrossEntropy" in args.criterion or args.criterion == "ce":
        criterion = nn.CrossEntropyLoss()
    elif "mse" in args.criterion:
        criterion = nn.MSELoss()

    optimizer = None
    if optimizer_type == "Adam":
        optimizer = optim.Adam(
            model.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=1e-4
        )
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
    if model_type != "ae":
        save_model(
            model_save_path,
            model_type,
            trianing_result,
            best_valid_loss,
            model,
            dataloader.dataset.classes,
        )
    else:
        pass
