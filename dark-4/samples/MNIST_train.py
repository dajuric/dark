import os
import pickle
import numpy as np
import dark
import dark.nn as nn
from dark.nn.init import default_init_weights
from dark.optim import *
from dark.utils.data import ImageFolder, DataLoader
from dark.utils.transforms import *

IM_SIZE = 28
BATCH_SIZE = 32
CLASS_COUNT = 10
EPOCHS = 5
model_path = "samples/model.pickle"

class MyNNBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.linear = nn.Linear(in_dim, out_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        return x

class MyNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()
        self.block1 = MyNNBlock(IM_SIZE*IM_SIZE, 512)
        self.block2 = MyNNBlock(512, 512)
        self.linear = nn.Linear(512, 10)

    def forward(self, x):
        x = self.flatten(x)

        x = self.block1(x)
        x = self.block2(x)

        x = self.linear(x)
        return x


def get_loaders():
    def label_transform(l):
        one_hot = np.zeros(CLASS_COUNT, dtype=np.float32)
        one_hot[l] = 1
        return one_hot

    trTransforms = Compose(
        Resize(IM_SIZE, IM_SIZE),
        Grayscale(),
        FlipHorizontal(),
        Normalize(0.5, 0.5),
        ToTensorV2()
    )

    teTransforms = Compose(
        Resize(IM_SIZE, IM_SIZE),
        Grayscale(),
        Normalize(0.5, 0.5),
        ToTensorV2()
    )

    trSet = ImageFolder("samples/db-FashionMNIST/train/", trTransforms, label_transform)
    trLoader = DataLoader(trSet, BATCH_SIZE, shuffle=True)

    teSet = ImageFolder("samples/db-FashionMNIST/test/", teTransforms, label_transform)
    teLoader = DataLoader(teSet, BATCH_SIZE)

    return trLoader, teLoader

def get_net():
    net = None
    if os.path.exists(model_path):
        net = pickle.load(open(model_path, "rb")) 
    else:
        net = MyNN()
        net.apply(default_init_weights)

    return net

def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    size = len(dataloader.dataset)
    correct = 0.0

    for batchIdx, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        correct += (pred.value.argmax(1) == y.argmax(1)).astype(np.float32).sum().item()

        if batchIdx % 100 == 0:
            loss, current = loss.value.item(), batchIdx * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    correct /= size
    print(f"Train: \n  Accuracy: {(100*correct):>0.1f}% \n") 

#@dark.no_grad()
def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    for X, y in dataloader:
        pred = model(X)
        test_loss += loss_fn(pred, y).value.item()
        correct += (pred.value.argmax(1) == y.argmax(1)).astype(np.float32).sum().item()

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
            pickle.dump(model, open(model_path, "wb"))

    print("Done!")

if __name__ == "__main__":
    np.seterr(over='raise')
    main()