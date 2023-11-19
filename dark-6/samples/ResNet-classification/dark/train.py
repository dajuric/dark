import os
import pickle
import dark
import dark.nn as nn
from dark.nn.init import default_init_weights
from dark.optim import *
from dark.utils.data import ImageFolder, DataLoader
from dark.utils.transforms import *
import dark.tensor as dt
from rich.progress import track
from model import Resnet9

IM_SIZE = 32
BATCH_SIZE = 128
CLASS_COUNT = 2 # 10 for full dataset
EPOCHS = 5

print(f"Running on: {'cuda' if dt.is_cuda() else 'cpu'}")
script_dir = os.path.dirname(os.path.realpath(__file__))
model_path = f"{script_dir}/model.pickle"

def get_loaders():
    def label_transform(l):
        one_hot = dt.zeros(CLASS_COUNT, dtype=dt.float64)
        one_hot[l] = 1
        return one_hot

    trTransforms = Compose(
        Resize(IM_SIZE, IM_SIZE),
        FlipHorizontal(),
        Normalize(0.5, 0.5),
        ToTensorV2()
    )

    teTransforms = Compose(
        Resize(IM_SIZE, IM_SIZE),
        Normalize(0.5, 0.5),
        ToTensorV2()
    )

    trSet = ImageFolder(f"{script_dir}/../db/train/", trTransforms, label_transform)
    trLoader = DataLoader(trSet, BATCH_SIZE, shuffle=True)

    teSet = ImageFolder(f"{script_dir}/../db/test/", teTransforms, label_transform)
    teLoader = DataLoader(teSet, BATCH_SIZE)

    return trLoader, teLoader

def get_net():
    net = None
    if os.path.exists(model_path):
        net = pickle.load(open(model_path, "rb")) 
    else:
        net = Resnet9(CLASS_COUNT)
        net.apply(default_init_weights)

    return net

def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    size = len(dataloader.dataset)
    correct = 0.0

    for batchIdx, (X, y) in enumerate(track(dataloader, "Training")):
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

def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    for X, y in track(dataloader, "Testing"):
        pred = model(X)
        test_loss += loss_fn(pred, y).data.item()
        correct += (pred.data.argmax(1) == y.argmax(1)).astype(dt.float32).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test: \n  Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
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
        test_loss = test_loop(te_loader, model, loss_fn)

        if test_loss < min_test_loss:
            min_test_loss = test_loss

            print(f"Saving best model\n\n")
            #pickle.dump(model, open(model_path, "wb"))

    print("Done!")

if __name__ == "__main__":
    main()