import numpy as np
import torch.nn as nn
from torch.optim import *
from config import *
from rich.progress import track
from model import MyNN, get_net
from dataset import get_loaders
from util import save_samples

def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    size = len(dataloader.dataset)
    correct = 0.0

    for batchIdx, (X, y) in enumerate(track(dataloader, "Training...")):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        correct += (pred.argmax(1) == y.argmax(1)).type(torch.float32).sum().item()

        if batchIdx % 100 == 0:
            loss, current = loss.item(), batchIdx * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    correct /= size
    print(f"Train: \n  Accuracy: {(100*correct):>0.1f}% \n") 

def test_loop(dataloader, model, loss_fn, epoch):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    for X, y in track(dataloader, "Testing... "):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        test_loss += loss_fn(pred, y).item()
        correct += (pred.argmax(1) == y.argmax(1)).type(torch.float32).sum().item()

    test_loss /= num_batches
    correct /= size

    print(f"Test: \n  Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    save_samples(dataloader.dataset, model, f"{script_dir}/results-{epoch}.png")
    return test_loss

def main():
    tr_loader, te_loader = get_loaders()
    model = get_net()
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=1e-3, momentum=0.9)
    min_test_loss = float("inf")

    for e in range(EPOCHS):
        print(f"Epoch {e+1}\n-------------------------------")
        train_loop(tr_loader, model, loss_fn, optimizer)
        test_loss = test_loop(te_loader, model, loss_fn, e)

        if test_loss < min_test_loss:
            min_test_loss = test_loss

            print(f"Saving best model\n\n")
            #torch.save(model, model_path)

    print("Done!")

if __name__ == "__main__":
    np.seterr(over='raise')
    main()