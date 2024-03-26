import argparse

from data import get_dataloader, default_transform, canny_transform
from model import ResNetClassifier
from utils import reset_seeds, save_model, evaluate_model

from train import train_model

import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Subset, random_split
from torch.utils.data import DataLoader

if __name__=="__main__":

  parser = argparse.ArgumentParser(description="Unzip a file.")

  # Add arguments
  parser.add_argument("--batch_size", type=int, default=64)
  parser.add_argument("--epochs", type=int, default=50)
  parser.add_argument("--lr", type=float, default=0.001)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--transform", type=str, choices=['default_transform', 'canny_transform'], default='default_transform')

  parser.add_argument("--model", type=str, choices=['resnet50', 'ViT'], default='resnet50', help="not implemented")
  parser.add_argument("--train_data_path", type=str, help="Path to load data")
  parser.add_argument("--model_save_path", type=str, help="Path to save the model's weight")
  

  # Parse arguments
  args = parser.parse_args()

  batch_size = args.batch_size
  epochs = args.epochs
  model_type = args.model
  model_save_path = args.model_save_path
  num_classes = 121
  train_data_path = args.train_data_path
  lr=args.lr
  seed = args.seed

  transform = default_transform
  if(args.transform == 'canny_transform'):
    transform = canny_transform
     
  reset_seeds(seed)

  """## data loading ##"""
  dataloader = get_dataloader(train_data_path, batch_size, True, transform)
  validation_size = int(0.2 * len(dataloader.dataset))
  train_dataset, val_dataset = random_split(dataloader.dataset, [len(dataloader.dataset) - validation_size, validation_size])

  # Create new DataLoaders for the training and validation datasets
  train_loader = DataLoader(train_dataset, batch_size=dataloader.batch_size, shuffle=True, num_workers=dataloader.num_workers)
  val_loader = DataLoader(val_dataset, batch_size=dataloader.batch_size, shuffle=False, num_workers=dataloader.num_workers)


  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = ResNetClassifier(num_classes=num_classes)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

  """##training##"""
  trianing_result, best_model_params = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=epochs, device='cuda')
  save_model(model_save_path, model_type, trianing_result, best_model_params, dataloader.dataset.classes)