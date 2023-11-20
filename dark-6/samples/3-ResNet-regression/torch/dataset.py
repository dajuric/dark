from torch.utils.data import Dataset
import json
import numpy as np
import cv2

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