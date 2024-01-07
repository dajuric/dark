import os
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Running on: {device}")

script_dir = os.path.dirname(os.path.realpath(__file__))
model_path = f"{script_dir}/model.pth"

IM_SIZE = 28
BATCH_SIZE = 32
CLASS_COUNT = 10
EPOCHS = 5