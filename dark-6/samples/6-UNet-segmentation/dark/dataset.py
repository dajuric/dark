from dark.utils.data import Dataset, DataLoader
import numpy as np
import random
from datetime import datetime
import cv2
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
            seed  =  datetime.now().timestamp()

            random.seed(seed)
            image = self.transform(image)

            random.seed(seed)
            mask  = self.transform(mask)

        return image, mask
    
def get_loaders():
    trSet = CarvanaDataset(f"{script_dir}/../db/train/", trTransforms)
    trLoader = DataLoader(trSet, BATCH_SIZE, shuffle=True)

    teSet = CarvanaDataset(f"{script_dir}/../db/train/", teTransforms)
    teLoader = DataLoader(teSet, BATCH_SIZE)

    return trLoader, teLoader


if __name__ == "__main__":
    from dark.utils.transforms import *
    import cv2

    trSet = CarvanaDataset(f"{script_dir}/../db/train/")
    for i, (im, mask) in enumerate(trSet):
        cv2.imwrite(f"{script_dir}/img-{i}.png", im)
        cv2.imwrite(f"{script_dir}/msk-{i}.png", mask)

        if i > 5: break