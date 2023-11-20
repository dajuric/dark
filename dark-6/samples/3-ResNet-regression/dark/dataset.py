from dark.utils.data import Dataset
import json
import numpy as np
import cv2
import random
from datetime import datetime

class KeypointDataset(Dataset):
    def __init__(self, kp_files, im_transform, kp_transform):
        self.kp_files = kp_files
        self.im_transform = im_transform
        self.kp_transform = kp_transform

    def __len__(self):
        return len(self.kp_files)

    def __getitem__(self, idx):
        seed = datetime.now().timestamp()
        kp_file = self.kp_files[idx]
        im_file = kp_file.replace(".json", ".png")

        kp_data = json.load(open(kp_file, "r"))
        x, y, w, h = kp_data["bbox"]
        kps = kp_data["bbox_keypoints"]
        kps = np.array(kps, dtype=np.float32).reshape(-1, 2)

        img = cv2.imread(im_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img[y:y+h, x:x+w, :]

        random.seed(seed)
        kps, _ = self.kp_transform(kps, img.shape)
        random.seed(seed)
        img = self.im_transform(img)

        return img, kps.reshape(-1)
    

if __name__ == "__main__":
    import dark.utils.transforms as T
    import dark.utils.point_transforms as P
    from glob import glob
    import os
    
    def _draw_keypoints(im, keypoints, color):
        imH, imW, _ = im.shape

        for x, y in keypoints.reshape(-1, 2):
            x = int(x * imW)
            y = int(y * imH)

            cv2.circle(im, (x, y), 3, color, -1)

    def show_sample(dataset, index):
        im, target = dataset[index]

        im = (im * 127 + 127).astype(np.uint8)
        im = cv2.cvtColor(np.moveaxis(im, 0, 2), cv2.COLOR_RGB2BGR)
        im = np.ascontiguousarray(im)

        _draw_keypoints(im, target, (0, 255, 0))
        cv2.imshow("Image", im)
        cv2.waitKey()


    script_dir = os.path.dirname(os.path.realpath(__file__))
    IM_SIZE = 128

    tr_im_transforms = T.Compose(   
        T.Resize(IM_SIZE, IM_SIZE),
        T.Rotate(limit=90, p=1),
        T.GaussianBlur(kernel_size=(3, 7), sigma_limit=(0.01, 1.5), p=1),
        T.BrightnessJitter(brightness=(-0.2, 0.2), p=1),
        T.ContrastJitter(contrast=(-0.5, 0.5), p=1),
        T.Normalize(0.5, 0.5),
        T.ToTensorV2(),
    )
    
    tr_pt_transforms = P.Compose(
        P.Resize(IM_SIZE, IM_SIZE),
        P.Rotate(limit=90, p=1),
        P.Normalize()
    )

    kp_files = sorted(glob(f"{script_dir}/../db/images/*.json"))
    trSet = KeypointDataset(kp_files, tr_im_transforms, tr_pt_transforms)

    show_sample(trSet, 0)
