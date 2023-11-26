import os
import dark.tensor as dt
from dark.utils.transforms import *

IM_SIZE = 64
BATCH_SIZE = 24
EPOCHS = 3

print(f"Running on: {'cuda' if dt.is_cuda() else 'cpu'}")
script_dir = os.path.dirname(os.path.realpath(__file__))
model_path = f"{script_dir}/model.pth"

trTransforms = Compose(
    Resize(IM_SIZE, IM_SIZE),
    Rotate(limit=35, p=1.0),
    FlipHorizontal(p=0.5),
    FlipVertical(p=0.1),
    Normalize(0.0, 1.0),
    ToTensorV2()
)

teTransforms = Compose(
    Resize(IM_SIZE, IM_SIZE),
    Normalize(0.0, 1.0),
    ToTensorV2()
)