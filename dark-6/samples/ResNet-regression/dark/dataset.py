from dark.utils.data import Dataset
import json
import numpy as np
import cv2
import random

class KeypointDataset(Dataset):
    def __init__(self, kp_files, im_transform, kp_transform):
        self.kp_files = kp_files
        self.im_transform = im_transform
        self.kp_transform = kp_transform

    def __len__(self):
        return len(self.kp_files)

    def __getitem__(self, idx):
        kp_file = self.kp_files[idx]
        im_file = kp_file.replace(".json", ".png")

        kp_data = json.load(open(kp_file, "r"))
        x, y, w, h = kp_data["bbox"]
        kps = kp_data["bbox_keypoints"]
        kps = np.array(kps, dtype=np.float32).reshape(-1, 2)

        img = cv2.imread(im_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img[y:y+h, x:x+w, :]

        random.seed(1)
        img = self.im_transform(img)
        random.seed(1)
        kps = self.kp_transform(kps)

        return img, kps.reshape(-1)