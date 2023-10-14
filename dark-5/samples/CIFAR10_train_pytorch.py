import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import *
from torchvision.datasets import ImageFolder
from torchvision.transforms import *

IM_SIZE = 32
BATCH_SIZE = 64
CLASS_COUNT = 3 # 10 for full dataset
EPOCHS = 5
model_path = "samples/model.pt"

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=0)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        out = self.pool(out)
        return out

class MyConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            ConvBlock(3, 6),
            ConvBlock(6, 16),
            nn.Flatten(),

            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),

            nn.Linear(120, 84),
            nn.ReLU(),

            nn.Linear(84, CLASS_COUNT)
        )

    def forward(self, x):
        logits = self.network(x)
        return logits


def get_loaders():
    def label_transform(l):
        one_hot = np.zeros(CLASS_COUNT, dtype=np.float64)
        one_hot[l] = 1
        return one_hot

    trTransforms = Compose([
        Resize(IM_SIZE),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(0.5, 0.5),
    ])

    teTransforms = Compose([
        Resize(IM_SIZE),
        ToTensor(),
        Normalize(0.5, 0.5),
    ])

    trSet = ImageFolder("samples/db-CIFAR10/train/", trTransforms, label_transform)
    trLoader = DataLoader(trSet, BATCH_SIZE, shuffle=True)

    teSet = ImageFolder("samples/db-CIFAR10/test/", teTransforms, label_transform)
    teLoader = DataLoader(teSet, BATCH_SIZE)

    return trLoader, teLoader

def get_net():
    net = None
    if os.path.exists(model_path):
        net = torch.load(model_path) 
    else:
        net = MyConvNet()

    return net

def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    size = len(dataloader.dataset)
    correct = 0.0

    for batchIdx, (X, y) in enumerate(dataloader):
        optimizer.zero_grad()
        
        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        
        correct += (pred.argmax(1) == y.argmax(1)).type(torch.float32).sum().item()

        if batchIdx % 100 == 0:
            loss, current = loss.item(), batchIdx * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    correct /= size
    print(f"Train: \n  Accuracy: {(100*correct):>0.1f}% \n") 

@torch.no_grad()
def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    for X, y in dataloader:
        pred = model(X)
        test_loss += loss_fn(pred, y).item()
        correct += (pred.argmax(1) == y.argmax(1)).type(torch.float32).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test: \n  Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss

def main():
    tr_loader, te_loader = get_loaders()
    model = get_net()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=1e-2, momentum=0.9)
    min_test_loss = float("inf")

    for e in range(EPOCHS):
        print(f"Epoch {e+1}\n-------------------------------")
        train_loop(tr_loader, model, loss_fn, optimizer)
        test_loss = test_loop(te_loader, model, loss_fn)

        if test_loss < min_test_loss:
            min_test_loss = test_loss

            print(f"Saving best model\n\n")
            torch.save(model, model_path)

    print("Done!")

if __name__ == "__main__":
    main()