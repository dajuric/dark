import dark.tensor as dt
import albumentations as A
import cv2
import os
import numpy as np

script_dir = os.path.dirname(os.path.realpath(__file__))

DB_PATH = f"{script_dir}/../db/TCDCN/"
MODEL_PATH = f"{script_dir}/model.pt"
C = 1
TEST_SIZE = 0.1

IM_SIZE = 128

ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)]
] 
NUM_ANCHORS = 3
for sa in ANCHORS: assert NUM_ANCHORS == len(sa)

S = [IM_SIZE // 32, IM_SIZE // 16]

BATCH_SIZE = 64
LEARNING_RATE = 1e-3
NUM_EPOCHS = 50

print(f"Running on: {'cuda' if dt.is_cuda() else 'cpu'}")

class ToTensorV2():
    def __init__(self):
        super().__init__()

    def __call__(self, **kwargs):
        im = kwargs["image"]

        if im.ndim < 3: im = np.expand_dims(im, -1)      
        im = np.rollaxis(im, -1, 0) #channels first

        out = { "image": im, "bboxes": np.array(kwargs["bboxes"]) }
        return out


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



