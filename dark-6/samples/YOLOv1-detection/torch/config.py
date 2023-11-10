import torch

torch.manual_seed(0)
torch.autograd.set_detect_anomaly(True)
device = "cuda" if torch.cuda.is_available() else "cpu"

IM_SIZE = 448
BATCH_SIZE = 16
S = 7
C = 20

LEARNING_RATE = 2e-5
EPOCHS = 250