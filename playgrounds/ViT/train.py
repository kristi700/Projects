import torch
import torchvision.transforms as transforms

from data.datasets import CIFAR10Dataset
from torch.utils.data import DataLoader, random_split, Subset

def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(112),
    ])

    cifar_dataset = CIFAR10Dataset('cifar-10/trainLabels.csv', 'cifar-10/train', transform=transform)

    total_size = len(cifar_dataset)
    val_size = int(total_size * 0.2)
    train_size = total_size - val_size

    train_data, val_data = random_split(cifar_dataset,[train_size, val_size])

    train_data_loader = DataLoader(train_data, 128)
    val_data_loader = DataLoader(val_data, 128)


if __name__ == "__main__":
    main()