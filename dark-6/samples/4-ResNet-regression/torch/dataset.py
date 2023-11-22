from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import json
import numpy as np
import cv2
import albumentations as A
import albumentations.pytorch.transforms as AT
from config import *
import random
from glob import glob

class KeypointDataset(Dataset):
    def __init__(self, kp_files, transform, tgt_im_size):
        self.kp_files = kp_files
        self.transform = transform
        self.tgt_im_size = tgt_im_size

    def __len__(self):
        return len(self.kp_files)

    def __getitem__(self, idx):
        kp_file = self.kp_files[idx]
        im_file = kp_file.replace(".json", ".png")

        kp_data = json.load(open(kp_file, "r"))
        x, y, w, h = kp_data["bbox"]
        kps = np.array(kp_data["bbox_keypoints"], dtype=np.float32)

        img = cv2.imread(im_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        img = img[y:y+h, x:x+w, :]
        data = self.transform(image=img, keypoints=kps.reshape(-1, 2))
       
        img = data["image"]
        kps = np.array(data["keypoints"], dtype=np.float32).reshape(-1)
        kps /= self.tgt_im_size
       
        return img, kps
    

def get_loaders():
    trTransforms = A.Compose([  
        A.Resize(IM_SIZE, IM_SIZE),
        A.Rotate(limit=90),
        A.GaussianBlur(blur_limit=(3, 7), sigma_limit=(0.01, 1.5)),
        A.ColorJitter(brightness=0.2, contrast=0.2, hue=0, saturation=0),
        A.Normalize(0.5, 0.5),   
        AT.ToTensorV2()
    ],
    keypoint_params = A.KeypointParams(format='xy', remove_invisible=False))
     
    teTransforms = A.Compose([  
        A.Resize(IM_SIZE, IM_SIZE),
        A.Normalize(0.5, 0.5),   
        AT.ToTensorV2()
    ],
    keypoint_params = A.KeypointParams(format='xy', remove_invisible=False))

    kp_files = sorted(glob(f"{script_dir}/../db/images/*.json"))
    random.shuffle(kp_files)

    tr_portion = 0.8
    tr_files_count = int(tr_portion * len(kp_files))

    trSet = KeypointDataset(kp_files[:tr_files_count], trTransforms, IM_SIZE)
    trLoader = DataLoader(trSet, BATCH_SIZE, shuffle=True)

    teSet = KeypointDataset(kp_files[tr_files_count:], teTransforms, IM_SIZE)
    teLoader = DataLoader(teSet, BATCH_SIZE)

    return trLoader, teLoader