from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import *
from config import *

def get_loader():
    tr = Compose([
        Resize((64, 64)),
        ToTensor(),
        Normalize(0.5, 0.5)
    ])

    dataset = ImageFolder(f"{script_dir}/../db/", transform=tr)
    dataloader = DataLoader(dataset, BATCH_SIZE, shuffle=True, drop_last=True)
    return dataloader
