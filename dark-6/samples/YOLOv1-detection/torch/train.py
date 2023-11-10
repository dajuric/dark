from torch.optim import *
from config import *
from dataset import *
from model import *
from loss import *
from utils import *
from tqdm import tqdm

def train_loop(dLoader: DataLoader, model: YoloV1, loss_fn: YoloLoss, optimizer: Optimizer):
    model.train()
    losses = []

    for X, y in dLoader:
        X, y = X.to(device), y.to(device)
        pred = model(X)

        loss = loss_fn(pred, y)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    mean_loss = sum(losses) / len(losses)
    print(f"Mean train loss: {mean_loss}")

@torch.no_grad()
def test_loop(dLoader: DataLoader, model: YoloV1, loss_fn: YoloLoss):
    model.eval()
    losses = []

    for X, y in dLoader:
        X, y = X.to(device), y.to(device)
        pred = model(X)

        loss = loss_fn(pred, y)
        losses.append(loss.item())

    mean_loss = sum(losses) / len(losses)
    print(f"Mean test loss: {mean_loss}")
    return mean_loss

def main():
    model = YoloV1().to(device)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = YoloLoss()
    trLoader, teLoader = get_dataloaders()

    min_test_loss = float("inf")
    for epoch in range(EPOCHS):
        print(f"\n-----Epoch: {epoch}-----")
        train_loop(trLoader, model, loss_fn, optimizer)
        test_loss = test_loop(teLoader, model, loss_fn)

        if test_loss < min_test_loss:
            torch.save(model, "model.pt")
            min_test_loss = test_loss


if __name__ == "__main__":
    main()



