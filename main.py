import argparse

from data import get_dataloader, default_transform, canny_transform
from model import ResNetClassifier, ViTClassifier
from utils import reset_seeds, save_model

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
  parser.add_argument("--beta1", type=float, default=0.9, help = "decay rate for the first moment estimate for Adam")
  parser.add_argument("--beta2", type=float, default=0.999, help="exponential decay rate for Adam")
  parser.add_argument("--momentum", type=float, default=0.1, help="momentum for SGD")
  parser.add_argument("--optimizer_type", type=str, choices=['SGD', 'Adam'], default='Adam')
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--transform", type=str, choices=['default_transform', 'canny_transform'], default='default_transform')

  parser.add_argument("--model", type=str, choices=['resnet50', 'ViT'], default='resnet50', help="not implemented")
  parser.add_argument("--train_data_path", type=str, help="Path to load data")
  parser.add_argument("--model_save_path", type=str, help="Path to save the model's weight")
  

  # Parse arguments
  args = parser.parse_args()

  batch_size = args.batch_size
  epochs = args.epochs

  optimizer_type = args.optimizer_type
  lr=args.lr
  beta1 = args.beta1
  beta2 = args.beta2
  momentum = args.momentum
  
  model_type = args.model
  model_save_path = args.model_save_path
  train_data_path = args.train_data_path
  
  seed = args.seed
  num_classes = 121
     
  reset_seeds(seed)


  """## data loading ##"""
  transform = default_transform
  if(args.transform == 'canny_transform'):
    transform = canny_transform

  dataloader = get_dataloader(train_data_path, batch_size, True, transform)
  validation_size = int(0.2 * len(dataloader.dataset))
  train_dataset, val_dataset = random_split(dataloader.dataset, [len(dataloader.dataset) - validation_size, validation_size])

  # Create new DataLoaders for the training and validation datasets
  train_loader = DataLoader(train_dataset, batch_size=dataloader.batch_size, shuffle=True, num_workers=dataloader.num_workers)
  val_loader = DataLoader(val_dataset, batch_size=dataloader.batch_size, shuffle=False, num_workers=dataloader.num_workers)

  """##models, loss and optimizer set up"""
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
  model = None
  if model_type=="resnet50":
    model = ResNetClassifier(num_classes=num_classes)
  else:
    model = ViTClassifier(num_classes=num_classes)

  criterion = nn.CrossEntropyLoss()

  optimizer = None
  if optimizer_type == "Adam":
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2))
  else:
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

  """##training##"""
  trianing_result, best_model_params = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=epochs, device='cuda')
  save_model(model_save_path, model_type, trianing_result, best_model_params, dataloader.dataset.classes)