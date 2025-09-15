import time
import os
import sys
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
import numpy as np

def measure_imagenet_io_speed(imagenet_root, split='train', batch_size=64, num_workers=4):
    # Define image transform (no resize/crop to focus on I/O)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()  # Converts PIL image to tensor [0,1]
    ])

    # Use ImageFolder assuming the structure matches standard ImageNet format
    dataset_path = os.path.join(imagenet_root, split)
    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    total_bytes = 0
    start_time = time.time()

    for images, _ in dataloader:
        # Measure the memory footprint of the loaded tensor
        total_bytes += images.element_size() * images.nelement()

    end_time = time.time()
    duration = end_time - start_time
    total_gb = total_bytes / (1024 ** 3)
    speed_gbps = total_gb / duration

    print(f"[Split={split}, num_workers={num_workers}] Time: {duration:.2f}s | Total: {total_gb:.3f} GB | Speed: {speed_gbps:.3f} GB/s")

if __name__ == "__main__":
    imagenet_root = sys.argv[1]  # "/orcd/datasets/001/imagenet/images_complete/ilsvrc"

#    for workers in [0, 2, 4, 8, 16]:
    for workers in [96, 64, 32, 16, 8, 4, 2, 0]:
        measure_imagenet_io_speed(imagenet_root, split='train', batch_size=128, num_workers=workers)

