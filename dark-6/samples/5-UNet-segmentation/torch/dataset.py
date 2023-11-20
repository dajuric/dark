from torch.utils.data import Dataset
import pickle
import numpy as np
import random
from datetime import datetime

class CarvanaDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        fn = 'train64' if train else 'test64'
        with open(f'{root}/{fn}', 'rb') as f:
            data = pickle.load(f)

        self.images = (data['images'] / 255).astype(np.float32)
        self.masks = data['masks'].astype(np.float32)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index].reshape(64, 64, 3)
        mask = self.masks[index].reshape(64, 64)

        if self.transform is not None:
            seed  =  datetime.now().timestamp()

            random.seed(seed)
            image = self.transform(image)

            random.seed(seed)
            mask  = self.transform(mask)

        return image, mask