import os
import dark.tensor as dt
import dark.utils.transforms as T
import dark.utils.point_transforms as P

print(f"Running on: {'cuda' if dt.is_cuda() else 'cpu'}")
script_dir = os.path.dirname(os.path.realpath(__file__))
model_path = f"{script_dir}/model.pth"

IM_SIZE = 96
KEYPOINT_COUNT = 68 * 2
BATCH_SIZE = 64
EPOCHS = 15


tr_im_transforms = T.Compose(   
    T.Resize(IM_SIZE, IM_SIZE),
    T.Rotate(limit=90),
    T.GaussianBlur(kernel_size=(3, 7), sigma_limit=(0.01, 1.5)),
    T.BrightnessJitter(brightness=(-0.2, 0.2)),
    T.ContrastJitter(contrast=(-0.2, 0.2)),
    T.Normalize(0.5, 0.5),
    T.ToTensorV2(),
)

tr_pt_transforms = P.Compose(
    P.Resize(IM_SIZE, IM_SIZE),
    P.Rotate(limit=90),
    P.Normalize()
)


te_im_transforms = T.Compose(
    T.Resize(IM_SIZE, IM_SIZE),
    T.Normalize(0.5, 0.5),
    T.ToTensorV2(),
)

te_kp_transforms = P.Compose(
    P.Resize(IM_SIZE, IM_SIZE),
    P.Normalize()
)