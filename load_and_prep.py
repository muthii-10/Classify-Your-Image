#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torchvision import datasets,transforms
from input_args import get_input_args
import os

args = get_input_args()

def load_prep_data(data_dir):

    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')
    test_dir = os.path.join(data_dir, 'test')
    
    
    # Define your transforms for the training, validation, and testing sets
    train_data_transforms = transforms.Compose([
        transforms.Resize(256),  # Resize 
        transforms.CenterCrop(224),  # Crop to 224x224
        transforms.RandomRotation(30),  # Add random rotation for data augmentation
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])

    valid_data_transforms = transforms.Compose([
        transforms.Resize(256),  # Resize 
        transforms.CenterCrop(224),  # Center crop to 224x224
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Normalize using same means and stds
    ])

    test_data_transforms = transforms.Compose([
        transforms.Resize(256),  # Resize 
        transforms.CenterCrop(224),  # Center crop to 224x224
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Normalize using same means and stds
    ])

    # Load the datasets with ImageFolder
    train_dataset = datasets.ImageFolder(train_dir, transform=train_data_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_data_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_data_transforms)

    # Define the dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, valid_loader, test_loader