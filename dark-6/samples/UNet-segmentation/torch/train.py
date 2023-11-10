import os
import pickle
import numpy as np
import dark
import dark.nn as nn
from dark.nn.init import default_init_weights
from dark.optim import *
from dark.utils.data import DataLoader
from dark.utils.transforms import *
import dark.tensor as dt
from rich.progress import track
from dataset import CarvanaDataset

IM_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 5
model_path = "samples/model.pickle"

print(f"Running on: {'cuda' if dt.is_cuda() else 'cpu'}")
script_dir = os.path.dirname(os.path.realpath(__file__))

from model import UNet

import cv2
import numpy as np

def test_samples(dataset, model, filename, count = 5):
    image_rows = []

    for i in range(count):
        im, true_mask = dataset[i]
                
        im, true_mask = dt.asarray(im), dt.asarray(true_mask)
        im = dt.expand_dims(im, 0)

        pred_mask = model(im).data
        den = dt.add(1, dt.exp(dt.negative(pred_mask)))
        pred_mask = dt.divide(1, den)

        im        = (im.squeeze()        * 255).astype(dt.uint8)
        pred_mask = (pred_mask.squeeze() * 255).astype(dt.uint8)
        true_mask = (true_mask.squeeze() * 255).astype(dt.uint8)

        row = dt.concatenate([im, dt.stack([true_mask] * 3, axis=0), dt.stack([pred_mask] * 3, axis=0)], axis=1)
        row = dt.rollaxis(row, 0, 3)
        image_rows.append(row)

    image_table = dt.concatenate(image_rows, axis=1)
    cv2.imwrite(filename, image_table.get())


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

    trSet = CarvanaDataset(f"{script_dir}/../db/", True, trTransforms)
    trLoader = DataLoader(trSet, BATCH_SIZE, shuffle=True)

    teSet = CarvanaDataset(f"{script_dir}/../db/", False, teTransforms)
    teLoader = DataLoader(teSet, BATCH_SIZE)

    return trLoader, teLoader

def get_net():
    net = None
    if os.path.exists(model_path):
        net = pickle.load(open(model_path, "rb")) 
    else:
        net = UNet(3, 1)
        net.apply(default_init_weights)

    return net

def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    size = len(dataloader.dataset)

    for batchIdx, (X, y) in enumerate(track(dataloader, "Training")):
        optimizer.zero_grad()
        
        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        
        if batchIdx % 10 == 0:
            loss, current = loss.data.item(), batchIdx * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn, epoch):
    model.eval()
    num_batches = len(dataloader)
    test_loss = 0

    for X, y in track(dataloader, "Testing"):
        pred = model(X)
        test_loss += loss_fn(pred, y).data.item()

    test_loss /= num_batches
    print(f"Test: \n  Avg loss: {test_loss:>8f} \n")

    test_samples(dataloader.dataset, model, f"{script_dir}/samples-{epoch}.png", 5)
    return test_loss

def main():
    tr_loader, te_loader = get_loaders()
    model = get_net()
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=1e-3)
    min_test_loss = float("inf")

    for e in range(EPOCHS):
        print(f"Epoch {e+1}\n-------------------------------")
        train_loop(tr_loader, model, loss_fn, optimizer)
        test_loss = test_loop(te_loader, model, loss_fn, e)

        if test_loss < min_test_loss:
           min_test_loss = test_loss

           print(f"Saving best model\n\n")
           pickle.dump(model, open(model_path, "wb"))

    print("Done!")

if __name__ == "__main__":
    main()