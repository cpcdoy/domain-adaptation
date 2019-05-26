from torchvision import datasets, transforms

import torch
from torch import nn

def load_datasets(path, train=True):
    """
    Load datasets for both training or evaluating the model.
    
    Downloads the datasets if they're not on disk.
    
    Parameters:
    -path (str): Path to the datasets
	-train (bool, default=True): Gets either train or test datasets
    
	Returns:
	-A dict with datasets for both source and target
    """
    
    # Resize both dataset samples to 32x32x3
    img_size = 32
    
    # Apply a few transform such as resizing, color jittering and normalization with mean and std
    transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ColorJitter(.1, 1, .75, 0),    
            transforms.ToTensor(),
            transforms.Lambda(lambda x : x.expand([3, -1, -1])),
            transforms.Normalize(mean=(0.1307, 0.1307, 0.1307), std=(0.3081 ,0.3081 ,0.3081))
    ])
    mnist = datasets.MNIST(path, train=train, download=True, transform=transform)
    
    # Apply a few transform such as resizing and normalization with mean and std
    transform = transforms.Compose([
            transforms.Resize(img_size),   
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.45, 0.45, 0.45), std=(0.199,0.199, 0.199))
    ])
    svhn = datasets.SVHN(path, split='train' if train else 'test', download=True, transform=transform)
    
    return {'mnist' : mnist, 'svhn' : svhn}

