import torch
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
script_dir = os.path.dirname(os.path.realpath(__file__))
modelD_path = f"{script_dir}/modelD.pth"
modelG_path = f"{script_dir}/modelG.pth"

nz = 100
BATCH_SIZE = 64
EPOCHS = 15