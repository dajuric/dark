import torch
from torch.utils.data import Dataset, DataLoader
from config import *
from glob import glob
from pathlib import Path
import cv2

class CarvanaDataset(Dataset):
    def __init__(self, im_folder, transform=None):
        self.im_files = sorted(glob(f"{im_folder}/*.jpg"))
        self.transform = transform

    def __len__(self):
        return len(self.im_files)

    def __getitem__(self, index):
        im_file = self.im_files[index]

        folder = os.path.dirname(im_file)
        file   = Path(im_file).stem
        mask_file = f"{folder}/{file}_mask.png" 

        image = cv2.imread(im_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_file)
        if len(mask.shape) == 3: mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        if self.transform is not None:
            data = self.transform(image=image, mask=mask)
            image = data["image"]
            mask = data["mask"]

        mask = mask.type(torch.float32) / 255
        return image, mask
    

def get_loaders():
    trSet = CarvanaDataset(f"{script_dir}/../db/train/", trTransforms)
    trLoader = DataLoader(trSet, BATCH_SIZE, shuffle=True)

    teSet = CarvanaDataset(f"{script_dir}/../db/train/", teTransforms)
    teLoader = DataLoader(teSet, BATCH_SIZE)

    return trLoader, teLoader