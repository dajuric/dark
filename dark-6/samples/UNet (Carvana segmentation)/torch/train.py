import os
import pickle
import numpy as np
import tortto
import tortto.nn as nn
from tortto.optim import *
from rich.progress import track
from dataset import CarvanaDataset
from dark.utils.data import DataLoader
from dark.utils.transforms import *

IM_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 5
model_path = "samples/model.pickle"

from model import UNet

def get_loaders():
    trTransforms = Compose(
        Rotate(limit=35, p=1.0),
        FlipHorizontal(p=0.5),
        FlipVertical(p=0.1),
        ToTensorV2()
    )

    teTransforms = Compose(
        ToTensorV2()
    )

    trSet = CarvanaDataset("samples/db-Carvana/", True, trTransforms)
    trLoader = DataLoader(trSet, BATCH_SIZE, shuffle=True)

    teSet = CarvanaDataset("samples/db-Carvana/", False, teTransforms)
    teLoader = DataLoader(teSet, BATCH_SIZE)

    return trLoader, teLoader

def get_net():
    net = None
    if os.path.exists(model_path):
        net = pickle.load(open(model_path, "rb")) 
    else:
        net = UNet(3, 1)
        net.apply(nn.init.default_init_weights)

    net = net.cuda()
    return net

def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    size = len(dataloader.dataset)

    for batchIdx, (X, y) in enumerate(track(dataloader, "Training")):
        X = tortto.tensor(X).cuda(); y = tortto.tensor(y).cuda()
        optimizer.zero_grad()
        
        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        
        if batchIdx % 10 == 0:
            loss, current = loss.item(), batchIdx * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    model.eval()
    num_batches = len(dataloader)
    test_loss = 0

    for X, y in track(dataloader, "Testing"):
        X = tortto.tensor(X).cuda(); y = tortto.tensor(y).cuda()

        pred = model(X)
        test_loss += loss_fn(pred, y).item()

    test_loss /= num_batches
    print(f"Test: \n  Avg loss: {test_loss:>8f} \n")
    return test_loss

def main():
    tr_loader, te_loader = get_loaders()
    model = get_net()
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=1e-3) #SGD(model.parameters(), lr=1e-3, momentum=0.9)
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