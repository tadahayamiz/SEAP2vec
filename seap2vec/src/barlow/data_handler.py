# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:09:08 2019

data handler
TransformsをBT用に変更する

@author: tadahaya
"""
import torch
import torchvision
import torchvision.transforms as transforms

from PIL import Image, ImageOps, ImageFilter
import torchvision.transforms as transforms
import random

import numpy as np
from typing import Tuple

# ToDo BT用に変更
# frozen
class MyDataset(torch.utils.data.Dataset):
    """ to create my dataset """
    def __init__(self, input=None, output=None, transform=None):
        if input is None:
            raise ValueError('!! Give input !!')
        if output is None:
            raise ValueError('!! Give output !!')
        if type(transform) == list:
            if len(transform) != 0:
                if transform[0] is None:
                    self.transform = []
                else:
                    self.transform = transform
            else:
                self.transform = transform
        else:
            if transform is None:
                self.transform = []
            else:
                self.transform = [transform]
        self.input = input
        self.output = output
        self.datanum = len(self.input)

    def __len__(self):
        return self.datanum

    def __getitem__(self, idx):
        input = self.input[idx]
        output = self.output[idx]
        if len(self.transform) > 0:
            for t in self.transform:
                input = t(input)
        return input, output


class MyTransforms:
    def __init__(self) -> None:
        pass

    def __call__(self, x: np.float) -> torch.Tensor:
        x = torch.from_numpy(x.astype(np.float32))  # example
        return x


def prep_dataset(input, output, transform=None) -> torch.utils.data.Dataset:
    """
    prepare dataset from row data
    
    Parameters
    ----------
    data: array
        input data such as np.array

    label: array
        input labels such as np.array
        would be None with unsupervised learning

    transform: a list of transform functions
        each function should return torch.tensor by __call__ method
    
    """
    return MyDataset(input, output, transform)


def prep_dataloader(
    dataset, batch_size, shuffle=None, num_workers=2, pin_memory=True
    ) -> torch.utils.data.DataLoader:
    """
    prepare train and test loader
    
    Parameters
    ----------
    dataset: torch.utils.data.Dataset
        prepared Dataset instance
    
    batch_size: int
        the batch size
    
    shuffle: bool
        whether data is shuffled or not

    num_workers: int
        the number of threads or cores for computing
        should be greater than 2 for fast computing
    
    pin_memory: bool
        determines use of memory pinning
        should be True for fast computing
    
    """
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_worker_init_fn
        )    
    return loader


def _worker_init_fn(worker_id):
    """ fix the seed for each worker """
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def prep_data(
    train_x, train_y, test_x, test_y, batch_size,
    transform=(None, None), shuffle=(True, False),
    num_workers=2, pin_memory=True
    ) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    prepare train and test loader from data
    combination of prep_dataset and prep_dataloader for model building
    
    Parameters
    ----------
    train_x, train_y, test_x, test_y: arrays
        arrays for training data, training labels, test data, and test labels
    
    batch_size: int
        the batch size

    transform: a tuple of transform functions
        transform functions for training and test, respectively
        each given as a list
    
    shuffle: (bool, bool)
        indicates shuffling training data and test data, respectively
    
    num_workers: int
        the number of threads or cores for computing
        should be greater than 2 for fast computing
    
    pin_memory: bool
        determines use of memory pinning
        should be True for fast computing    

    """
    train_dataset = prep_dataset(train_x, train_y, transform[0])
    test_dataset = prep_dataset(test_x, test_y, transform[1])
    train_loader = prep_dataloader(
        train_dataset, batch_size, shuffle[0], num_workers, pin_memory
        )
    test_loader = prep_dataloader(
        test_dataset, batch_size, shuffle[1], num_workers, pin_memory
        )
    return train_loader, test_loader


def prepare_test(
        batch_size=4, num_workers=2, train_sample_size=None, test_sample_size=None,
        train_transform=(None, None), test_transform=(None, None)
        ):
    if train_transform[0] is None:
        train_transform = Transform()
    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=train_transform
    )
    if train_sample_size is not None:
        indices = torch.randperm(len(trainset))[:train_sample_size]
        trainset = torch.utils.data.Subset(trainset, indices)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
    if test_transform[0] is None:
        test_transform = Transform()
    testset = torchvision.datasets.CIFAR10(
        root="./", train=False, download=True, transform=test_transform
        )
    if test_sample_size is not None:
        indices = torch.randperm(len(testset))[:test_sample_size]
        testset = torch.utils.data.Subset(testset, indices)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    classes = (
        'plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
        )
    return trainloader, testloader, classes


class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class Transform:
    def __init__(self, transform=None, transform_prime=None):
        '''

        :param transform: Transforms to be applied to first input
        :param transform_prime: transforms to be applied to second
        '''
        if transform == None:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                            saturation=0.2, hue=0.1)],
                    p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(p=1.0),
                Solarization(p=0.0),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
        if transform_prime == None:

            self.transform_prime = transforms.Compose([
                transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                            saturation=0.2, hue=0.1)],
                    p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(p=0.1),
                Solarization(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform_prime = transform_prime

    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
        return y1, y2