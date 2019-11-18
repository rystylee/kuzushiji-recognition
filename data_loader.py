import os
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from dataset import KMNIST49


def load_dataset(data_root, dataset_name, trans):
    if dataset_name == 'mnist':
        train_dataset = torchvision.datasets.MNIST(
            root=data_root,
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]))
        test_dataset = torchvision.datasets.MNIST(
            root=data_root,
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]))
    elif dataset_name == 'kmnist49':
        train_dataset = KMNIST49(
            root=data_root,
            train=True,
            download=True
        )
        test_dataset = KMNIST49(
            root=data_root,
            train=False,
            download=True
        )
    else:
        train_dataset = datasets.ImageFolder(
            root=os.path.join(data_root, dataset_name, 'train'),
            transform=trans)
        test_dataset = datasets.ImageFolder(
            root=os.path.join(data_root, dataset_name, 'test'),
            transform=trans)
    return train_dataset, test_dataset


class DataLoader(object):
    def __init__(self, data_root, dataset_name, batch_size):
        self.data_root = data_root
        self.dataset_name = dataset_name
        self.batch_size = batch_size

        self.transforms = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])

    def get_loader(self):
        train_dataset, test_dataset = load_dataset(
            self.data_root, self.dataset_name, self.transforms)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, num_workers=4)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.batch_size, num_workers=4)

        print(f'Total number of train data: {len(train_loader.dataset)}')
        print(f'Total number of test data: {len(test_loader.dataset)}')
        print(f'Total number of classes: {len(train_dataset.classes)}\n')
        return train_loader, test_loader
