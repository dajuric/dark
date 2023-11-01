import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import *
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import *
from rich.progress import track
from model import MyConvNet

IM_SIZE = 32
BATCH_SIZE = 128
CLASS_COUNT = 2 # 10 for full dataset
EPOCHS = 5

print(f"Running on: {'cuda' if torch.cuda.is_available() else 'cpu'}")

def get_loaders():
    def label_transform(l):
        one_hot = torch.zeros(CLASS_COUNT, dtype=torch.float64)
        one_hot[l] = 1
        return one_hot

    trTransforms = Compose([
        Resize(IM_SIZE),
        Grayscale(),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(0.5, 0.5)
    ])

    teTransforms = Compose([
        Resize(IM_SIZE),
        Grayscale(),
        ToTensor(),
        Normalize(0.5, 0.5)
    ])

    trSet = ImageFolder("samples/db-CIFAR10/train/", trTransforms, label_transform)
    trLoader = DataLoader(trSet, BATCH_SIZE, shuffle=True)

    teSet = ImageFolder("samples/db-CIFAR10/test/", teTransforms, label_transform)
    teLoader = DataLoader(teSet, BATCH_SIZE)

    return trLoader, teLoader

def get_net():
    net = None
    if os.path.exists(model_path):
        net = torch.load(open(model_path, "rb")) 
    else:
        net = MyConvNet(CLASS_COUNT)
        net = net.cuda()

    return net

def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    size = len(dataloader.dataset)
    correct = 0.0

    for batchIdx, (X, y) in enumerate(track(dataloader, "Training")):
        X = X.cuda(); y = y.cuda()
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

    for X, y in track(dataloader, "Testing"):
        X = X.cuda(); y = y.cuda()

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
            #pickle.dump(model, open(model_path, "wb"))

    print("Done!")

if __name__ == "__main__":
    main()