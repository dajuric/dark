from pandas.core.frame import DataFrame
from scipy.sparse.construct import rand
import torch
from tqdm import tqdm
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
import albumentations as A
from torch.utils.data import Dataset, DataLoader
from torch.nn import *
import numpy as np
import timm

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using: {device}")

class FacialKeyDataset(Dataset):
    def __init__(self, df: DataFrame, imgFolder, augmentations = None):
        self.df = df
        self.imgFolder = imgFolder
        self.augmentations = augmentations

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        imgName = self.df.iloc[idx, 0]
        img = cv2.imread(os.path.join(self.imgFolder, imgName))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        keypoints = self.df.iloc[idx, 1:].to_numpy().reshape(-1, 2)
        if self.augmentations:
            ag_data = self.augmentations(image=img, keypoints=keypoints)
            img = torch.from_numpy(ag_data["image"]).float()
            keypoints = torch.tensor(ag_data["keypoints"]).float()

        return img.permute(2, 0, 1), keypoints.view(-1)

class FacialKeyModel(Module):

    def __init__(self, modelName="resnet18"):
        super(FacialKeyModel, self).__init__()
        self.backbone = timm.create_model(modelName, pretrained=True, num_classes=68*2)

    def forward(self, images, key=None):
        predictedKey = self.backbone(images)

        if key is not None: 
            return predictedKey, MSELoss()(predictedKey, key)
        return predictedKey

def train_loop(model: Module, trainLoader: DataLoader, optimizer: torch.optim.Adam):
    trainLoss = 0.0
    model.train()

    for data in tqdm(trainLoader):
        images, keys = data
        images, keys = images.to(device), keys.to(device)

        predictedKeys, loss = model(images, keys)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        trainLoss += loss.item()

    return trainLoss / len(trainLoader)

def eval_loop(model: Module, validLoader: DataLoader):
    validLoss = 0.0
    model.eval()

    with torch.no_grad():
        for data in tqdm(validLoader):
            images, keys = data
            images, keys = images.to(device), keys.to(device)

            predictedKeys, loss = model(images, keys)

            validLoss += loss.item()

    return validLoss / len(validLoader)


def showItem(im, k, denormalize=True):
    im = im.permute(1, 2, 0).cpu().detach().numpy()
    if denormalize:
        im = im * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])  #from Normalize() augmentation im * std + mean

    keypoints = k.reshape(-1, 2).cpu().detach().numpy()
    plt.imshow(im)
    plt.scatter(keypoints[:, 0], keypoints[:, 1], s=4, c='r')

    plt.show()

def showItems(im, kA, kB, denormalize=True):
    im = im.permute(1, 2, 0).cpu().detach().numpy()
    if denormalize:
        im = im * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])  #from Normalize() augmentation im * std + mean

    kA = kA.reshape(-1, 2).cpu().detach().numpy()
    kB = kB.reshape(-1, 2).cpu().detach().numpy()

    plt.subplot(1, 2, 1)
    plt.imshow(im)
    plt.scatter(kA[:, 0], kA[:, 1], s=8, c='b')

    plt.subplot(1, 2, 2)
    plt.imshow(im)
    plt.scatter(kB[:, 0], kB[:, 1], s=8, c='r')

    plt.show()


if __name__ == "__main__":
    #------------ train
    train_df = pd.read_csv("data/train_frames_keypoints.csv")

    augs = A.Compose([
        A.Resize(140, 140),
        A.Normalize()
    ], 
    keypoint_params = A.KeypointParams(format='xy', remove_invisible=False))

    train_df, valid_df = train_test_split(train_df, test_size=0.2, random_state=42)
    trainset = FacialKeyDataset(train_df, "data/train/", augs); #showItem(*trainset[23])
    validset = FacialKeyDataset(valid_df, "data/train/", augs)

    trainLoader = DataLoader(trainset, batch_size=32, shuffle=False)
    validLoader = DataLoader(validset, batch_size=32, shuffle=False)
            
    model = FacialKeyModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    bestValidLoss = np.Inf
    for e in range(0, 50):
        avgTrainLoss = train_loop(model, trainLoader, optimizer)
        avgValidLoss = eval_loop(model, validLoader)

        if avgValidLoss < bestValidLoss:
            torch.save(model, "model/FacialModel.pth")
            bestValidLoss = avgValidLoss

        print(f"Epoch: {e}, avgTrainLoss: {avgTrainLoss}")
        print(f"Epoch: {e}, avgValidLoss: {avgValidLoss}")


