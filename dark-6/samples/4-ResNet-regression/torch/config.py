import os
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using: {device}")

script_dir = os.path.dirname(os.path.realpath(__file__))
model_path = f"{script_dir}/model.pth"

IM_SIZE = 96
KEYPOINT_COUNT = 68 * 2
BATCH_SIZE = 64
EPOCHS = 20