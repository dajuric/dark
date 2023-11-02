import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision.transforms import *
import torch.optim as optim

import os
import random
from glob import glob
from rich.progress import track

from model import Resnet9
from dataset import KeypointDataset
from util import save_samples


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using: {device}")

script_dir = os.path.dirname(os.path.realpath(__file__))
model_path = f"{script_dir}/model.pth"

IM_SIZE = 128
KEYPOINT_COUNT = 14 * 2
BATCH_SIZE = 64
EPOCHS = 10

def get_loaders():
    trTransforms = Compose([   
        ToTensor(),
        Resize((IM_SIZE, IM_SIZE), antialias=None),
        Normalize(0.5, 0.5)
    ])

    teTransforms = Compose([
        ToTensor(),
        Resize((IM_SIZE, IM_SIZE), antialias=None),
        Normalize(0.5, 0.5)
    ])

    kp_files = sorted(glob(f"{script_dir}/../db/images/*.json"))
    random.shuffle(kp_files)

    tr_portion = 0.8
    tr_files_count = int(tr_portion * len(kp_files))

    trSet = KeypointDataset(kp_files[:tr_files_count], trTransforms)
    trLoader = DataLoader(trSet, BATCH_SIZE, shuffle=True)

    teSet = KeypointDataset(kp_files[tr_files_count:], teTransforms)
    teLoader = DataLoader(teSet, BATCH_SIZE)

    return trLoader, teLoader

def get_net():
    net = None
    if os.path.exists(model_path):
        net = torch.load(open(model_path, "rb")) 
    else:
        net = Resnet9(KEYPOINT_COUNT)
        net = net.cuda()

    return net


def train_loop(train_loader, model, criterion, optimizer):
    train_loss = 0.0
    model.train()

    for data in track(train_loader, "Training"):
        ims, kps = data
        ims, kps = ims.to(device), kps.to(device)

        predicted_kps = model(ims)
        loss = criterion(predicted_kps, kps)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss = train_loss / len(train_loader)
    print(f"Train: loss: {(train_loss * 100):>0.1f}") 
    return train_loss

@torch.no_grad()
def test_loop(val_loader, model, criterion, epoch):
    val_loss = 0.0
    model.eval()

    for data in track(val_loader, "Testing"):
        ims, kps = data
        ims, kps = ims.to(device), kps.to(device)

        predicted_kps = model(ims)
        loss = criterion(predicted_kps, kps)

        val_loss += loss.item()

    val_loss = val_loss / len(val_loader)
    print(f"Eval: loss: {(val_loss * 100):>0.1f}") 

    save_samples(val_loader.dataset, model, f"{script_dir}/samples-{epoch}.png", device=device)
    return val_loss


def main():
    tr_loader, te_loader = get_loaders()
    model = get_net()
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3) #optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
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


