# -*- coding: utf-8 -*-
"""Copy of AI6102.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/14P6VfChelYs1JyID4vQAQer5aBfQa-Ve

## data loading##

## Training of the model ##
parse parameters
"""

import numpy as np
import torch
import copy
import random
import os
import pdb


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler=None, num_epochs=10, device='cuda'):
  model.to(device)
  trianing_result = []
  best_model_params = copy.deepcopy(model.state_dict())
  best_valid_loss = 999999
  for epoch in range(num_epochs):
    # Training phase
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
      images, labels = images.to(device), labels.to(device)

      #train the model
      optimizer.zero_grad()
      outputs = model(images)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      #saving loss and acc
      running_loss += loss.item() * images.size(0)
      _, predicted = torch.max(outputs, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()


    #calculate loss and accuracy
    train_loss = running_loss / len(train_loader.dataset)
    train_acc = correct / total
    print(f"Epoch : {epoch}")
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

    val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)

    epoch_result = {}
    epoch_result["Epoch"] = epoch
    epoch_result["Train Loss"] = train_loss
    epoch_result["Val Loss"] = val_loss
    epoch_result["Train Acc"] = train_acc
    epoch_result["Val Acc"] = val_acc
    trianing_result.append(epoch_result)

    if val_loss < best_valid_loss:
      best_valid_loss = val_loss
      best_model_params = copy.deepcopy(model.state_dict())

    #update scheduler if not None
    if(scheduler):
       scheduler.step()

  return trianing_result, best_model_params


def evaluate_model(model, val_loader, criterion, device='cuda'):
  model.eval()
  val_loss = 0.0
  correct = 0
  total = 0
  with torch.no_grad():
      for images, labels in val_loader:
          images, labels = images.to(device), labels.to(device)
          outputs = model(images)
          loss = criterion(outputs, labels)
          val_loss += loss.item() * images.size(0)
          _, predicted = torch.max(outputs, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()

  val_loss = val_loss / len(val_loader.dataset)
  val_acc = correct / total

  print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
  return val_loss, val_acc
