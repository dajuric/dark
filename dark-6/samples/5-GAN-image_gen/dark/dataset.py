from dark.utils.data import ImageFolder, DataLoader
from dark.utils.transforms import *
from config import *

def get_loader():
    tr = Compose(
        Resize(64 ,64),
        Normalize(0.5, 0.5),
        ToTensorV2(),
    )

    dataset = ImageFolder(f"{script_dir}/../db/", imgT=tr)
    data_loader = DataLoader(dataset, BATCH_SIZE, shuffle=True, drop_last=True)
    return data_loader