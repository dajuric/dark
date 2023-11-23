
from dark.utils.data import ImageFolder, DataLoader
from dark.utils.transforms import *
from config import *


def get_loaders():
    def label_transform(l):
        one_hot = np.zeros(CLASS_COUNT, dtype=np.float32)
        one_hot[l] = 1
        return one_hot

    trTransforms = Compose(
        Resize(IM_SIZE, IM_SIZE),
        Grayscale(),
        FlipHorizontal(),
        Normalize(0.5, 0.5),
        ToTensorV2()
    )

    teTransforms = Compose(
        Resize(IM_SIZE, IM_SIZE),
        Grayscale(),
        Normalize(0.5, 0.5),
        ToTensorV2()
    )

    trSet = ImageFolder(f"{script_dir}/../db/train/", trTransforms, label_transform)
    trLoader = DataLoader(trSet, BATCH_SIZE, shuffle=True)

    teSet = ImageFolder(f"{script_dir}/../db/test/", teTransforms, label_transform)
    teLoader = DataLoader(teSet, BATCH_SIZE)

    return trLoader, teLoader