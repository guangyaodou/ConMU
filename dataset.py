import copy

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, SVHN

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def cifar10_dataloaders_no_val(batch_size=128, data_dir='datasets/cifar10', num_workers=2):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    print('Dataset information: CIFAR-10\t 45000 images for training \t 5000 images for validation\t')
    print('10000 images for testing\t no normalize applied in data_transform')
    print('Data augmentation = randomcrop(32,4) + randomhorizontalflip')

    train_set = CIFAR10(data_dir, train=True,
                        transform=train_transform, download=True)
    val_set = CIFAR10(data_dir, train=False,
                      transform=test_transform, download=True)
    test_set = CIFAR10(data_dir, train=False,
                       transform=test_transform, download=True)

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader


def svhn_dataloaders(batch_size=128, data_dir='datasets/svhn', num_workers=2, class_to_replace: int = None,
                     num_indexes_to_replace=None, indexes_to_replace=None, seed: int = 1, only_mark: bool = False,
                     shuffle=True, no_aug=False):
    train_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    print('Dataset information: SVHN\t 45000 images for training \t 5000 images for validation\t')

    train_set = SVHN(data_dir, split="train",
                     transform=train_transform, download=True)

    test_set = SVHN(data_dir, split="test",
                    transform=test_transform, download=True)

    train_set.labels = np.array(train_set.labels)
    test_set.labels = np.array(test_set.labels)

    rng = np.random.RandomState(seed)
    valid_set = copy.deepcopy(train_set)
    valid_idx = []
    for i in range(max(train_set.labels) + 1):
        class_idx = np.where(train_set.labels == i)[0]
        valid_idx.append(rng.choice(class_idx, int(
            0.1 * len(class_idx)), replace=False))
    valid_idx = np.hstack(valid_idx)
    train_set_copy = copy.deepcopy(train_set)

    valid_set.data = train_set_copy.data[valid_idx]
    valid_set.labels = train_set_copy.labels[valid_idx]

    train_idx = list(set(range(len(train_set))) - set(valid_idx))

    train_set.data = train_set_copy.data[train_idx]
    train_set.labels = train_set_copy.labels[train_idx]

    if class_to_replace is not None and indexes_to_replace is not None:
        raise ValueError(
            "Only one of `class_to_replace` and `indexes_to_replace` can be specified")
    if class_to_replace is not None:
        replace_class(train_set, class_to_replace, num_indexes_to_replace=num_indexes_to_replace, seed=seed - 1,
                      only_mark=only_mark)
        if num_indexes_to_replace is None or num_indexes_to_replace == 4454:
            test_set.data = test_set.data[test_set.labels != class_to_replace]
            test_set.labels = test_set.labels[test_set.labels !=
                                              class_to_replace]

    if indexes_to_replace is not None:
        replace_indexes(dataset=train_set, indexes=indexes_to_replace,
                        seed=seed - 1, only_mark=only_mark)

    loader_args = {'num_workers': 0, 'pin_memory': False}

    def _init_fn(worker_id):
        np.random.seed(int(seed))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              worker_init_fn=_init_fn if seed is not None else None, **loader_args)
    val_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False,
                            worker_init_fn=_init_fn if seed is not None else None, **loader_args)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             worker_init_fn=_init_fn if seed is not None else None, **loader_args)

    return train_loader, val_loader, test_loader


def cifar100_dataloaders(batch_size=128, data_dir='datasets/cifar100', num_workers=2, class_to_replace: int = None,
                         num_indexes_to_replace=None, indexes_to_replace=None, seed: int = 1, only_mark: bool = False,
                         shuffle=True, no_aug=False):
    if no_aug:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    print('Dataset information: CIFAR-100\t 45000 images for training \t 500 images for validation\t')
    print('10000 images for testing\t no normalize applied in data_transform')
    print('Data augmentation = randomcrop(32,4) + randomhorizontalflip')
    train_set = CIFAR100(data_dir, train=True,
                         transform=train_transform, download=True)

    test_set = CIFAR100(data_dir, train=False,
                        transform=test_transform, download=True)
    train_set.targets = np.array(train_set.targets)
    test_set.targets = np.array(test_set.targets)

    rng = np.random.RandomState(seed)
    valid_set = copy.deepcopy(train_set)
    valid_idx = []
    for i in range(max(train_set.targets) + 1):
        class_idx = np.where(train_set.targets == i)[0]
        valid_idx.append(rng.choice(class_idx, int(
            0.1 * len(class_idx)), replace=False))
    valid_idx = np.hstack(valid_idx)
    train_set_copy = copy.deepcopy(train_set)

    valid_set.data = train_set_copy.data[valid_idx]
    valid_set.targets = train_set_copy.targets[valid_idx]

    train_idx = list(set(range(len(train_set))) - set(valid_idx))

    train_set.data = train_set_copy.data[train_idx]
    train_set.targets = train_set_copy.targets[train_idx]

    if class_to_replace is not None and indexes_to_replace is not None:
        raise ValueError(
            "Only one of `class_to_replace` and `indexes_to_replace` can be specified")
    if class_to_replace is not None:
        replace_class(train_set, class_to_replace, num_indexes_to_replace=num_indexes_to_replace, seed=seed - 1,
                      only_mark=only_mark)
        if num_indexes_to_replace is None:
            test_set.data = test_set.data[test_set.targets != class_to_replace]
            test_set.targets = test_set.targets[test_set.targets !=
                                                class_to_replace]
    if indexes_to_replace is not None or indexes_to_replace == 450:
        replace_indexes(dataset=train_set, indexes=indexes_to_replace,
                        seed=seed - 1, only_mark=only_mark)

    loader_args = {'num_workers': 0, 'pin_memory': False}

    def _init_fn(worker_id):
        np.random.seed(int(seed))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              worker_init_fn=_init_fn if seed is not None else None, **loader_args)
    val_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False,
                            worker_init_fn=_init_fn if seed is not None else None, **loader_args)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             worker_init_fn=_init_fn if seed is not None else None, **loader_args)

    return train_loader, val_loader, test_loader


def cifar100_dataloaders_no_val(batch_size=128, data_dir='datasets/cifar100', num_workers=2):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    print('Dataset information: CIFAR-100\t 45000 images for training \t 500 images for validation\t')
    print('10000 images for testing\t no normalize applied in data_transform')
    print('Data augmentation = randomcrop(32,4) + randomhorizontalflip')

    train_set = CIFAR100(data_dir, train=True,
                         transform=train_transform, download=True)
    val_set = CIFAR100(data_dir, train=False,
                       transform=test_transform, download=True)
    test_set = CIFAR100(data_dir, train=False,
                        transform=test_transform, download=True)

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader


def cifar10_dataloaders(batch_size=128, data_dir='datasets/cifar10', num_workers=2, class_to_replace: int = None,
                        num_indexes_to_replace=None, indexes_to_replace=None, seed: int = 1, only_mark: bool = False,
                        shuffle=True, no_aug=False):
    if no_aug:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    print('Dataset information: CIFAR-10\t 45000 images for training \t 5000 images for validation\t')
    print('10000 images for testing\t no normalize applied in data_transform')
    print('Data augmentation = randomcrop(32,4) + randomhorizontalflip')

    train_set = CIFAR10(data_dir, train=True,
                        transform=train_transform, download=True)

    test_set = CIFAR10(data_dir, train=False,
                       transform=test_transform, download=True)

    train_set.targets = np.array(train_set.targets)
    test_set.targets = np.array(test_set.targets)

    rng = np.random.RandomState(seed)
    valid_set = copy.deepcopy(train_set)
    valid_idx = []
    for i in range(max(train_set.targets) + 1):
        class_idx = np.where(train_set.targets == i)[0]
        valid_idx.append(rng.choice(class_idx, int(
            0.1 * len(class_idx)), replace=False))
    valid_idx = np.hstack(valid_idx)
    train_set_copy = copy.deepcopy(train_set)

    valid_set.data = train_set_copy.data[valid_idx]
    valid_set.targets = train_set_copy.targets[valid_idx]

    train_idx = list(set(range(len(train_set))) - set(valid_idx))

    train_set.data = train_set_copy.data[train_idx]
    train_set.targets = train_set_copy.targets[train_idx]

    if class_to_replace is not None and indexes_to_replace is not None:
        raise ValueError(
            "Only one of `class_to_replace` and `indexes_to_replace` can be specified")
    if class_to_replace is not None:
        replace_class(train_set, class_to_replace, num_indexes_to_replace=num_indexes_to_replace, seed=seed - 1,
                      only_mark=only_mark)
        if num_indexes_to_replace is None or num_indexes_to_replace == 4500:
            test_set.data = test_set.data[test_set.targets != class_to_replace]
            test_set.targets = test_set.targets[test_set.targets !=
                                                class_to_replace]
    if indexes_to_replace is not None:
        replace_indexes(dataset=train_set, indexes=indexes_to_replace,
                        seed=seed - 1, only_mark=only_mark)

    loader_args = {'num_workers': 0, 'pin_memory': False}

    def _init_fn(worker_id):
        np.random.seed(int(seed))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              worker_init_fn=_init_fn if seed is not None else None, **loader_args)
    val_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False,
                            worker_init_fn=_init_fn if seed is not None else None, **loader_args)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             worker_init_fn=_init_fn if seed is not None else None, **loader_args)

    return train_loader, val_loader, test_loader


def replace_indexes(dataset: torch.utils.data.Dataset, indexes, seed=0,
                    only_mark: bool = False):
    if not only_mark:
        rng = np.random.RandomState(seed)
        new_indexes = rng.choice(
            list(set(range(len(dataset))) - set(indexes)), size=len(indexes))
        dataset.data[indexes] = dataset.data[new_indexes]
        try:
            dataset.targets[indexes] = dataset.targets[new_indexes]
        except:
            dataset.labels[indexes] = dataset.labels[new_indexes]
        else:
            dataset._labels[indexes] = dataset._labels[new_indexes]
    else:
        # Notice the -1 to make class 0 work
        try:
            dataset.targets[indexes] = - dataset.targets[indexes] - 1
        except:
            try:
                dataset.labels[indexes] = - dataset.labels[indexes] - 1
            except:
                dataset._labels[indexes] = - dataset._labels[indexes] - 1


def replace_class(dataset: torch.utils.data.Dataset, class_to_replace: int, num_indexes_to_replace: int = None,
                  seed: int = 0, only_mark: bool = False):
    if class_to_replace == -1:
        try:
            indexes = np.flatnonzero(np.ones_like(dataset.targets))
        except:
            try:
                indexes = np.flatnonzero(np.ones_like(dataset.labels))
            except:
                indexes = np.flatnonzero(np.ones_like(dataset._labels))
    else:
        try:
            indexes = np.flatnonzero(
                np.array(dataset.targets) == class_to_replace)
        except:
            try:
                indexes = np.flatnonzero(
                    np.array(dataset.labels) == class_to_replace)
            except:
                indexes = np.flatnonzero(
                    np.array(dataset._labels) == class_to_replace)

    if num_indexes_to_replace is not None:
        assert num_indexes_to_replace <= len(
            indexes), f"Want to replace {num_indexes_to_replace} indexes but only {len(indexes)} samples in dataset"
        rng = np.random.RandomState(seed)
        indexes = rng.choice(
            indexes, size=num_indexes_to_replace, replace=False)
        print(f"Replacing indexes {indexes}")
    replace_indexes(dataset, indexes, seed, only_mark)


class DatasetFromSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


class CustomImageDataset(Dataset):
    def __init__(self, x, targets, transform=None):
        self.x = x
        self.targets = targets
        self.transform = transform

    def __getitem__(self, index):
        if self.transform:
            x = self.transform(self.x[index])
        else:
            x = self.x[index]

        return torch.tensor(x), torch.tensor(self.targets[index])

    def __len__(self):
        return len(self.targets)
