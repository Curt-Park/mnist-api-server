"""Train a simple conv net for MNIST dataset.

This script is a little modified from:
    https://github.com/pytorch/examples/blob/master/mnist/main.py

- Author: Jinwoo Park
- Email: www.jwpark.co.kr@gmail.com
"""

import argparse
import logging
import logging.config
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F
from model import ConvNet
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

logging.config.fileConfig("../../logging.conf")
logger = logging.getLogger()


def parse_args() -> argparse.Namespace:
    """Parse training settings."""
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1.0,
        metavar="LR",
        help="learning rate (default: 1.0)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        metavar="M",
        help="Learning rate step gamma (default: 0.7)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )
    args = parser.parse_args()
    return args


def get_dataloaders(
    batch_size: int, test_batch_size: int, use_cuda: bool
) -> Tuple[DataLoader, DataLoader]:
    """Get dataloaders for training and test."""
    # set kwargs for training and test
    cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
    train_kwargs: Dict[str, Any] = {"batch_size": batch_size}
    test_kwargs: Dict[str, Any] = {"batch_size": test_batch_size}
    train_kwargs.update(cuda_kwargs if use_cuda else {})
    test_kwargs.update(cuda_kwargs if use_cuda else {})

    # set dataset loaders
    transform_seq = [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    transform = transforms.Compose(transform_seq)
    data_path = "../../data"
    dataset1 = datasets.MNIST(data_path, train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST(data_path, train=False, transform=transform)
    train_loader = DataLoader(dataset1, **train_kwargs)
    test_loader = DataLoader(dataset2, **test_kwargs)

    return train_loader, test_loader


def train(
    model: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    params: Dict[str, int],
    device: torch.device,
) -> None:
    """Train the model in a single epoch."""
    model.train()
    epoch = params["epoch"]
    for data, target in tqdm(train_loader, desc=f"Train Epoch {epoch}"):
        data, target = data.to(device), target.to(device)

        # update
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()


def test(
    model: torch.nn.Module, test_loader: DataLoader, device: torch.device
) -> float:
    """Get the test accuracy."""
    model.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Evaluation"):
            data, target = data.to(device), target.to(device)

            # sum up batch loss
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()

            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc = 100.0 * correct / len(test_loader.dataset)

    logger.info(
        "Test set: Average loss: %.4f\tAccuracy: %d/%d (%.2f%%)",
        test_loss,
        correct,
        len(test_loader.dataset),
        test_acc,
    )

    return test_acc


def main() -> None:
    """Get the conv model with the best acc."""
    args = parse_args()

    torch.manual_seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        use_cuda=use_cuda,
    )

    model = ConvNet().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    best_acc = 0.0
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            params={"epoch": epoch, "log_interval": args.log_interval},
            device=device,
        )
        test_acc = test(model, test_loader, device)
        scheduler.step()

        if args.save_model and test_acc > best_acc:
            best_acc = test_acc
            model_scripted = torch.jit.script(model)
            model_scripted.save("../../mnist_cnn.pt")
            logger.info("The best model saved")


if __name__ == "__main__":
    main()
