"""Shared CIFAR-10 data loaders for Task 1 and Task 2."""

import os

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

try:
    from ..paths import DATA_ROOT
except ImportError:
    DATA_ROOT = os.path.normpath(
        os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data')
    )

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CIFAR10_ARCHIVE = 'cifar-10-python.tar.gz'
CIFAR10_ARCHIVE_BYTES = 170_498_071
CIFAR10_EXTRACTED_DIR = 'cifar-10-batches-py'


def _prepare_cifar10_root(root=DATA_ROOT):
    """Remove corrupted partial downloads before torchvision retries."""
    os.makedirs(root, exist_ok=True)
    archive_path = os.path.join(root, CIFAR10_ARCHIVE)
    extracted_path = os.path.join(root, CIFAR10_EXTRACTED_DIR)

    if os.path.isdir(extracted_path):
        return root

    if os.path.isfile(archive_path):
        size = os.path.getsize(archive_path)
        if size != CIFAR10_ARCHIVE_BYTES:
            os.remove(archive_path)
            print(
                f'Removed incomplete {CIFAR10_ARCHIVE} ({size} bytes). '
                f'Expected {CIFAR10_ARCHIVE_BYTES} bytes.'
            )
    return root


def _download_help(root=DATA_ROOT):
    archive_path = os.path.join(root, CIFAR10_ARCHIVE)
    return (
        'CIFAR-10 download failed. Delete any partial file and retry, or download manually:\n'
        '  https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz\n'
        f'Place the file here: {archive_path}\n'
        f'Or extract {CIFAR10_EXTRACTED_DIR}/ into: {root}'
    )


def _build_transforms(train: bool, augment: bool):
    if train and augment:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def _load_cifar10(root, train, augment):
    root = _prepare_cifar10_root(root)
    try:
        return datasets.CIFAR10(
            root=root,
            train=train,
            download=True,
            transform=_build_transforms(train, augment),
        )
    except RuntimeError as exc:
        archive_path = os.path.join(root, CIFAR10_ARCHIVE)
        if os.path.isfile(archive_path):
            os.remove(archive_path)
        raise RuntimeError(_download_help(root)) from exc


def get_cifar_loaders(
    root=DATA_ROOT,
    batch_size=128,
    num_workers=2,
    n_items=-1,
    augment_train=True,
):
    """Return train and test DataLoaders."""
    train_set = _load_cifar10(root, train=True, augment=augment_train)
    test_set = _load_cifar10(root, train=False, augment=False)

    if n_items > 0:
        train_set = Subset(train_set, range(min(n_items, len(train_set))))
        test_set = Subset(test_set, range(min(n_items, len(test_set))))

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader


def get_cifar_loader(
    root=DATA_ROOT,
    batch_size=128,
    train=True,
    shuffle=True,
    num_workers=2,
    n_items=-1,
    augment=None,
):
    """Single split loader (compatible with VGG_BatchNorm API)."""
    if augment is None:
        augment = train

    dataset = _load_cifar10(root, train=train, augment=augment)
    if n_items > 0:
        dataset = Subset(dataset, range(min(n_items, len(dataset))))

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and train,
        num_workers=num_workers,
        pin_memory=True,
    )


if __name__ == '__main__':
    import sys

    sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..')))

    import matplotlib.pyplot as plt
    import numpy as np

    loader = get_cifar_loader(augment=False)
    for images, labels in loader:
        print('label:', labels[0].item())
        print('shape:', images[0].shape)
        img = images[0].numpy().transpose(1, 2, 0)
        img = img * np.array(CIFAR10_STD) + np.array(CIFAR10_MEAN)
        img = np.clip(img, 0, 1)
        plt.imshow(img)
        plt.savefig('sample.png')
        print('Saved sample.png')
        break
