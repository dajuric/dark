from torch.optim import *
from torch.optim.lr_scheduler import *
from config import *
from dataset import *
from model import BlazeFace as YoloNet
from loss import *
from utils import *
from dataset import *
from rich.progress import track
import os

sAnchors = get_scaled_anchors().to(device)

def train_loop(dLoader: DataLoader, model: YoloNet, loss_fn: YoloLoss, optimizer: Optimizer):
    model.train()

    losses = []
    mean_loss = 0.0

    for X, y in track(dLoader, f"Train... [{mean_loss:3.2f}]"):
        X, y0, y1 = X.to(device), y[0].to(device), y[1].to(device)
        pred = model(X)

        loss = (
            loss_fn(pred[0], y0, sAnchors[0]) +
            loss_fn(pred[1], y1, sAnchors[1])
        ) 

        losses.append(loss.item())
        mean_loss = sum(losses) / len(losses)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Mean train loss: {mean_loss}")

@torch.no_grad()
def test_loop(dLoader: DataLoader, model: YoloNet, loss_fn: YoloLoss):
    model.eval()

    losses = []
    mean_loss = 0.0

    for X, y in track(dLoader, f"Eval...  [{mean_loss:3.2f}]"):
        X, y0, y1 = X.to(device), y[0].to(device), y[1].to(device)
        pred = model(X)

        loss = (
            loss_fn(pred[0], y0, sAnchors[0]) +
            loss_fn(pred[1], y1, sAnchors[1])
        )

        losses.append(loss.item())
        mean_loss = sum(losses) / len(losses)

    print(f"Mean test loss: {mean_loss}")
    save_detection_samples(model, dLoader.dataset, sAnchors)
    return mean_loss

def main():
    model = YoloNet().to(device)
    if os.path.exists(MODEL_PATH): model = torch.load(MODEL_PATH, map_location=device)

    loss_fn = YoloLoss()
    trLoader, teLoader = get_dataloaders()

    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True)

    min_test_loss = float("inf")
    for epoch in range(NUM_EPOCHS):
        print(f"\n-----Epoch: {epoch}-----")

        train_loop(trLoader, model, loss_fn, optimizer)
        test_loss = test_loop(teLoader, model, loss_fn)
        scheduler.step(test_loss)

        if test_loss < min_test_loss:
            torch.save(model, MODEL_PATH)
            min_test_loss = test_loss


if __name__ == "__main__":
    os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"
    np.seterr(over='raise')
    main()