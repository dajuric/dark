import os
import dark.tensor as dt

print(f"Running on: {'cuda' if dt.is_cuda() else 'cpu'}")
script_dir = os.path.dirname(os.path.realpath(__file__))
model_path = f"{script_dir}/model.pickle"

IM_SIZE = 28
BATCH_SIZE = 32
CLASS_COUNT = 10
EPOCHS = 5