import os
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
NUM_WORKERS = os.cpu_count()


def create_data_loader(train_dir: Path,
                       test_dir: Path,
                       transform: transforms.Compose,
                       batch_size: int = 32,
                       num_workers: int = NUM_WORKERS):
    
    # Use ImageFolder to create datasets and transform
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    test_data = datasets.ImageFolder(test_dir, transform=transform)

    # Create data loaders and get class names
    class_names = train_data.classes
    train_dl = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    test_dl = DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
    
    return train_dl, test_dl, class_names
