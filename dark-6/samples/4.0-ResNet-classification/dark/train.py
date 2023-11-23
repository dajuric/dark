import dark.nn as nn
from dark.optim import *
import dark.tensor as dt
from rich.progress import track
from model import Resnet9, get_net
from dataset import get_loaders
from config import *
from util import save_samples

def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    size = len(dataloader.dataset)
    correct = 0.0

    for batchIdx, (X, y) in enumerate(track(dataloader, "Training...")):
        optimizer.zero_grad()
        
        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        
        correct += (pred.data.argmax(1) == y.argmax(1)).astype(dt.float32).sum().item()

        if batchIdx % 10 == 0:
            loss, current = loss.data.item(), batchIdx * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    correct /= size
    print(f"Train: \n  Accuracy: {(100 * correct):>0.1f}% \n") 

def test_loop(dataloader, model, loss_fn, epoch):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    for X, y in track(dataloader, "Testing... "):
        pred = model(X)
        test_loss += loss_fn(pred, y).data.item()
        correct += (pred.data.argmax(1) == y.argmax(1)).astype(dt.float32).sum().item()

    test_loss /= num_batches
    correct /= size

    print(f"Test: \n  Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    save_samples(dataloader.dataset, model, dataloader.dataset.labels, f"{script_dir}/results-{epoch}.png")
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
            #pickle.dump(model, open(model_path, "wb"))

    print("Done!")

if __name__ == "__main__":
    main()