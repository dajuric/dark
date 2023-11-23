import os
import torch

IM_SIZE = 32
BATCH_SIZE = 128
CLASS_COUNT = 2 # 10 for full dataset
EPOCHS = 5


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Running on: {device}")

script_dir = os.path.dirname(os.path.realpath(__file__))
model_path = f"{script_dir}/model.pickle"