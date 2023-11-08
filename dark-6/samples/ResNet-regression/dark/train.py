import dark
import dark.tensor as dt
from dark.utils.data import DataLoader
import dark.nn as nn
import dark.optim as optim
import dark.utils.transforms as T
import dark.utils.point_transforms as P

import os
import random
from glob import glob
from rich.progress import track
import pickle

from model import Resnet18
from dataset import KeypointDataset
from util import save_samples


print(f"Running on: {'cuda' if dt.is_cuda() else 'cpu'}")
script_dir = os.path.dirname(os.path.realpath(__file__))
model_path = f"{script_dir}/model.pth"

IM_SIZE = 96
KEYPOINT_COUNT = 68 * 2
BATCH_SIZE = 64
EPOCHS = 10

def get_loaders():
    tr_im_transforms = T.Compose(   
        T.Resize(IM_SIZE, IM_SIZE),
        T.Rotate(limit=90),
        T.GaussianBlur(kernel_size=(3, 7), sigma_limit=(0.01, 1.5)),
        T.BrightnessJitter(brightness=(-0.2, 0.2)),
        T.ContrastJitter(contrast=(-0.2, 0.2)),
        T.Normalize(0.5, 0.5),
        T.ToTensorV2(),
    )
    
    tr_pt_transforms = P.Compose(
        P.Resize(IM_SIZE, IM_SIZE),
        P.Rotate(limit=90),
        P.Normalize()
    )
    

    te_im_transforms = T.Compose(
        T.Resize(IM_SIZE, IM_SIZE),
        T.Normalize(0.5, 0.5),
        T.ToTensorV2(),
    )
    
    te_kp_transforms = P.Compose(
        P.Resize(IM_SIZE, IM_SIZE),
        P.Normalize()
    )

    kp_files = sorted(glob(f"{script_dir}/../db/images/*.json"))
    random.shuffle(kp_files)

    tr_portion = 0.8
    tr_files_count = int(tr_portion * len(kp_files))

    trSet = KeypointDataset(kp_files[:tr_files_count], tr_im_transforms, tr_pt_transforms)
    trLoader = DataLoader(trSet, BATCH_SIZE, shuffle=True)

    teSet = KeypointDataset(kp_files[tr_files_count:], te_im_transforms, te_kp_transforms)
    teLoader = DataLoader(teSet, BATCH_SIZE)

    return trLoader, teLoader

def get_net():
    net = None
    if os.path.exists(model_path):
        net = pickle.load(open(model_path, "rb")) 
    else:
        net = Resnet18(out_dim=KEYPOINT_COUNT)

    return net


def train_loop(train_loader, model, criterion, optimizer):
    train_loss = 0.0
    model.train()

    for data in track(train_loader, "Training"):
        ims, kps = data

        predicted_kps = model(ims)
        loss = criterion(predicted_kps, kps)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.data.item()

    train_loss = train_loss / len(train_loader)
    print(f"Train: loss: {(train_loss * 100):>0.2f}") 
    
    save_samples(train_loader.dataset, model, f"{script_dir}/samples-train.png")
    return train_loss

def test_loop(val_loader, model, criterion, epoch):
    val_loss = 0.0
    model.eval()

    for data in track(val_loader, "Testing"):
        ims, kps = data

        predicted_kps = model(ims)
        loss = criterion(predicted_kps, kps)

        val_loss += loss.data.item()

    val_loss = val_loss / len(val_loader)
    print(f"Eval: loss: {(val_loss * 100):>0.2f}") 

    save_samples(val_loader.dataset, model, f"{script_dir}/samples-{epoch}.png")
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
            pickle.dump(model, open(model_path, "wb"))

    print("Done!")

if __name__ == "__main__":            
    main()


