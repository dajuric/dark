import dark
import dark.nn as nn
from dark.optim import *
from rich.progress import track
from dataset import get_loaders
from model import UNet, get_net
from util import save_samples
from config import *

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

    for X, y in track(dataloader, "Testing "):
        pred = model(X)
        test_loss += loss_fn(pred, y).data.item()

    test_loss /= num_batches
    print(f"Test: \n  Avg loss: {test_loss:>8f} \n")

    save_samples(dataloader.dataset, model, f"{script_dir}/samples-{epoch + 1}.png", 5)
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
           dark.save(model, model_path)

    print("Done!")

if __name__ == "__main__":
    main()