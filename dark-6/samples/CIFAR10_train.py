import os
import pickle
import numpy as np
import dark
import dark.nn as nn
from dark.nn.init import default_init_weights
from dark.optim import *
from dark.utils.data import ImageFolder, DataLoader
from dark.utils.transforms import *
import dark.tensor as dt


# im = dark.Parameter(dt.random.random((3, 6, 3)))
# out = dark.add(im, 1e-5)
# out.backward()
# exit()

# im = dark.Parameter(dt.random.random((6, 3, 160, 160)))
# bn1 = nn.BatchNorm2d(3)
# out = bn1(im)
# out = dark.sum(out)
# out.backward()
# exit()



IM_SIZE = 32
BATCH_SIZE = 8
CLASS_COUNT = 3 # 10 for full dataset
EPOCHS = 5
model_path = "samples/model.pickle"

print(f"Running on: {'cuda' if dt.is_cuda() else 'cpu'}")

# class ConvBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()

#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=0)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU()
#         self.pool = nn.MaxPool2d(2)

#     def forward(self, x):
#         x = self.conv(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.pool(x)
#         return x
    
# class MyConvNet(nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.network = nn.Sequential(
#             ConvBlock(3, 6),
#             ConvBlock(6, 16),
#             nn.Flatten(),

#             nn.Linear(16 * 5 * 5, 120),
#             nn.ReLU(),

#             nn.Linear(120, 84),
#             nn.ReLU(),

#             nn.Linear(84, CLASS_COUNT)
#         )

#     def forward(self, x):
#         logits = self.network(x)
#         return logits





# class MyConvNet(nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.network = nn.Sequential(
#             nn.Conv2d(3, 1, 3, 1),
#             nn.ReLU(),
#             nn.MaxPool2d(4),
#             nn.Flatten(),

#             nn.Linear(64, CLASS_COUNT),
#         )

#     def forward(self, x):
#         logits = self.network(x)
#         return logits








IM_SIZE = 160
from resnet9 import ResNet9 as MyConvNet

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

    trSet = ImageFolder("samples/db-CIFAR10/train/", trTransforms, label_transform)
    trLoader = DataLoader(trSet, BATCH_SIZE, shuffle=True)

    teSet = ImageFolder("samples/db-CIFAR10/test/", teTransforms, label_transform)
    teLoader = DataLoader(teSet, BATCH_SIZE)

    return trLoader, teLoader

def get_net():
    net = None
    if os.path.exists(model_path):
        net = pickle.load(open(model_path, "rb")) 
    else:
        net = MyConvNet()
        net.apply(default_init_weights)

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
        
        correct += (pred.value.argmax(1) == y.argmax(1)).astype(dt.float32).sum().item()

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
        correct += (pred.value.argmax(1) == y.argmax(1)).astype(dt.float32).sum().item()

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