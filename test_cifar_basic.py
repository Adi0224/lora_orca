#!/usr/bin/env python3
"""
Basic test to ensure CIFAR-100 loads and ORCA pipeline works
"""
import sys
sys.path.insert(0, 'ORCA/src')

import torch
import torchvision
import torchvision.transforms as transforms

print("Testing CIFAR-100 download and basic setup...")

# Test device detection
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

print(f"Device detected: {device}")

# Test CIFAR-100 download
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

print("Downloading CIFAR-100 train set...")
trainset = torchvision.datasets.CIFAR100(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

print("Downloading CIFAR-100 test set...")
testset = torchvision.datasets.CIFAR100(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

print(f"Train set size: {len(trainset)}")
print(f"Test set size: {len(testset)}")

# Test a single batch
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=4, shuffle=True, num_workers=0
)

dataiter = iter(trainloader)
images, labels = next(dataiter)
print(f"Batch shape: {images.shape}")
print(f"Labels shape: {labels.shape}")

print("\n✅ Basic setup works! CIFAR-100 is ready.")
