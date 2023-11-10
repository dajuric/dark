import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import os

script_dir = os.path.dirname(os.path.realpath(__file__))

DB_PATH = f"{script_dir}/../db/"
MODEL_PATH = f"{script_dir}/model.pt"
C = 1
TEST_SIZE = 0.1

IM_SIZE = 320

ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
] 
NUM_ANCHORS = 3
for sa in ANCHORS: assert NUM_ANCHORS == len(sa)

S = [IM_SIZE // 32, IM_SIZE // 16, IM_SIZE // 8]

BATCH_SIZE = 32
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 10

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

test_transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=IM_SIZE),
        A.PadIfNeeded(min_height=IM_SIZE, min_width=IM_SIZE, border_mode=cv2.BORDER_CONSTANT),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[]),
)

train_transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=IM_SIZE),
        A.PadIfNeeded(min_height=IM_SIZE, min_width=IM_SIZE, border_mode=cv2.BORDER_CONSTANT),
        
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.4),
        A.HorizontalFlip(p=0.5),
        A.Blur(p=0.1),
        A.CLAHE(p=0.1),

        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[]),
)



