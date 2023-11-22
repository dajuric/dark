import os
import torch
import albumentations as A
import albumentations.pytorch.transforms as AT

IM_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 3

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using: {device}")

script_dir = os.path.dirname(os.path.realpath(__file__))
model_path = f"{script_dir}/model.pth"


trTransforms = A.Compose([
    A.Rotate(limit=35, p=1.0),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.1),
    A.Normalize(0, 1),
    AT.ToTensorV2()
])

teTransforms = A.Compose([
    A.Normalize(0, 1),
    AT.ToTensorV2()
])
