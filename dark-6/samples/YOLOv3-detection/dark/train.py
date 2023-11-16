import dark
from dark.optim import *
from config import *
from dataset import *
from model import BlazeFace as YoloNet
from loss import *
from utils import *
from dataset import *
import os
import pickle

sAnchors = get_scaled_anchors()

def train_loop(dLoader: DataLoader, model: YoloNet, loss_fn: YoloLoss, optimizer: Optimizer):
    model.train()

    losses = []
    mean_loss = 0.0

    for X, y0, y1 in track(dLoader, lambda: f"Train... [{mean_loss:3.2f}]"):
        pred = model(X)

        loss = dark.add(
            loss_fn(pred[0], y0, sAnchors[0]),
            loss_fn(pred[1], y1, sAnchors[1])
        ) 

        losses.append(loss.data.item())
        mean_loss = sum(losses) / len(losses)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test_loop(dLoader: DataLoader, model: YoloNet, loss_fn: YoloLoss):
    model.eval()

    losses = []
    mean_loss = 0.0

    for X, y0, y1 in track(dLoader, lambda: f"Eval...  [{mean_loss:3.2f}]"):
        pred = model(X)

        loss = dark.add(
            loss_fn(pred[0], y0, sAnchors[0]),
            loss_fn(pred[1], y1, sAnchors[1])
        )

        losses.append(loss.data.item())
        mean_loss = sum(losses) / len(losses)

    save_detection_samples(model, dLoader.dataset, sAnchors)
    return mean_loss

def main():
    model = YoloNet()
    if os.path.exists(MODEL_PATH): model = pickle.load(open(MODEL_PATH, "rb"))

    loss_fn = YoloLoss()
    trLoader, teLoader = get_dataloaders()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    min_test_loss = float("inf")
    for epoch in range(NUM_EPOCHS):
        print(f"\n-----Epoch: {epoch}-----")

        train_loop(trLoader, model, loss_fn, optimizer)
        test_loss = test_loop(teLoader, model, loss_fn)

        if test_loss < min_test_loss:
            pickle.dump(model, open(MODEL_PATH, "wb"))
            min_test_loss = test_loss


if __name__ == "__main__":
    np.seterr(over='raise')
    main()
