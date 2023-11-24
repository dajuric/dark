import dark.tensor as dt
import os

print(f"Running on: {'cuda' if dt.is_cuda() else 'cpu'}")
script_dir = os.path.dirname(os.path.realpath(__file__))
modelD_path = f"{script_dir}/modelD.pickle"
modelG_path = f"{script_dir}/modelG.pickle"

nz = 100
BATCH_SIZE = 64
EPOCHS = 15