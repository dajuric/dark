from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import *
from config import *

def get_loaders():
    def label_transform(l):
        one_hot = torch.zeros(CLASS_COUNT, dtype=torch.float32)
        one_hot[l] = 1
        return one_hot

    trTransforms = Compose([
        Resize(IM_SIZE),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(0.5, 0.5)
    ])

    teTransforms = Compose([
        Resize(IM_SIZE),
        ToTensor(),
        Normalize(0.5, 0.5)
    ])

    trSet = ImageFolder(f"{script_dir}/../db/train/", trTransforms, label_transform)
    trLoader = DataLoader(trSet, BATCH_SIZE, shuffle=True)

    teSet = ImageFolder(f"{script_dir}/../db/test/", teTransforms, label_transform)
    teLoader = DataLoader(teSet, BATCH_SIZE)

    return trLoader, teLoader