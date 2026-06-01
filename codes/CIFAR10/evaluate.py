"""Evaluate a trained CIFARNet checkpoint."""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), '..')))

from common.utils.device import get_default_device, resolve_device, torch
import torch.nn as nn

from common.data.loaders import get_cifar_loaders
from models.cnn import CIFARNet, count_parameters


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate CIFARNet checkpoint')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--device', type=str, default=get_default_device())
    return parser.parse_args()


def main():
    args = parse_args()
    device = resolve_device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint.get('config', {})

    model = CIFARNet(
        width=config.get('width', 64),
        dropout=config.get('dropout', 0.5),
        activation=config.get('activation', 'relu'),
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    _, test_loader = get_cifar_loaders(batch_size=args.batch_size, num_workers=args.num_workers)
    test_loss, test_acc = evaluate(model, test_loader, device)
    result = {
        'checkpoint': args.checkpoint,
        'epoch': checkpoint.get('epoch'),
        'test_loss': test_loss,
        'test_acc': test_acc,
        'test_error': 1.0 - test_acc,
        'parameters': count_parameters(model),
    }

    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
