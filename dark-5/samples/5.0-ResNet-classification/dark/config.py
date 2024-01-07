import os
import dark.tensor as dt

IM_SIZE = 32
BATCH_SIZE = 128
CLASS_COUNT = 2 # 10 for full dataset
EPOCHS = 5

print(f"Running on: {'cuda' if dt.is_cuda() else 'cpu'}")
script_dir = os.path.dirname(os.path.realpath(__file__))
model_path = f"{script_dir}/model.pickle"