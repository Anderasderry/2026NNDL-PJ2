"""Train CIFARNet on CIFAR-10."""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), '..')))

from common.utils.device import get_default_device, resolve_device, set_seed, torch
from torch.optim import SGD, AdamW
from tqdm import tqdm

import torch.nn as nn

from common.data.loaders import get_cifar_loaders
from common.paths import CIFAR10_OUTPUT_DIR
from models.cnn import CIFARNet, count_parameters

OUTPUT_DIR = CIFAR10_OUTPUT_DIR
LOSS_FNS = {
    'ce': nn.CrossEntropyLoss,
    'label_smooth': lambda: nn.CrossEntropyLoss(label_smoothing=0.1),
}


def build_optimizer(name, params, lr, weight_decay):
    if name == 'sgd':
        return SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    if name == 'adamw':
        return AdamW(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f'Unknown optimizer: {name}')


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
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


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, leave=False, desc='train'):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def parse_args():
    parser = argparse.ArgumentParser(description='Train custom CNN on CIFAR-10')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--width', type=int, default=64, help='Base channel width')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--activation', choices=['relu', 'leaky_relu', 'gelu'], default='relu')
    parser.add_argument('--optimizer', choices=['sgd', 'adamw'], default='sgd')
    parser.add_argument('--loss', choices=list(LOSS_FNS.keys()), default='ce')
    parser.add_argument('--scheduler', choices=['cosine', 'step', 'none'], default='cosine')
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--n-items', type=int, default=-1, help='Use subset for quick debug')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--run-name', type=str, default='cifarnet')
    parser.add_argument('--device', type=str, default=get_default_device())
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    run_dir = os.path.join(OUTPUT_DIR, args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    device = resolve_device(args.device)
    train_loader, test_loader = get_cifar_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        n_items=args.n_items,
    )

    model = CIFARNet(
        width=args.width,
        dropout=args.dropout,
        activation=args.activation,
    ).to(device)

    criterion = LOSS_FNS[args.loss]()
    optimizer = build_optimizer(args.optimizer, model.parameters(), args.lr, args.weight_decay)

    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
    else:
        scheduler = None

    config = vars(args)
    config['parameters'] = count_parameters(model)
    with open(os.path.join(run_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)

    print(f'Device: {device}')
    print(f'Parameters: {config["parameters"]:,}')
    print(f'Run directory: {run_dir}')

    history = []
    best_acc = 0.0
    best_path = os.path.join(run_dir, 'best_model.pt')
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        if scheduler is not None:
            scheduler.step()

        lr = optimizer.param_groups[0]['lr']
        row = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'test_error': 1.0 - test_acc,
            'lr': lr,
        }
        history.append(row)

        print(
            f'Epoch {epoch:03d}/{args.epochs} | '
            f'train loss {train_loss:.4f} acc {train_acc:.4f} | '
            f'test loss {test_loss:.4f} acc {test_acc:.4f} err {1 - test_acc:.4f} | '
            f'lr {lr:.6f}'
        )

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'test_acc': test_acc,
                    'config': config,
                },
                best_path,
            )

    elapsed = time.time() - start
    history_path = os.path.join(run_dir, 'history.json')
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)

    summary = {
        'best_test_acc': best_acc,
        'best_test_error': 1.0 - best_acc,
        'epochs': args.epochs,
        'train_time_sec': elapsed,
        'parameters': config['parameters'],
        'checkpoint': best_path,
    }
    with open(os.path.join(run_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(f'\nBest test accuracy: {best_acc:.4f} (error {1 - best_acc:.4f})')
    print(f'Training time: {elapsed / 60:.1f} min')
    print(f'Saved checkpoint: {best_path}')
    print(f'History: {history_path}')


if __name__ == '__main__':
    main()
