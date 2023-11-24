from rich.progress import track
import dark
import dark.nn as nn
import dark.optim as optim
from model import Resnet18, get_net
from util import save_samples
from dataset import get_loaders
from config import *

def train_loop(train_loader, model, criterion, optimizer):
    train_loss = 0.0
    model.train()

    for batchIdx, data in enumerate(track(train_loader, "Training...")):
        ims, kps = data

        predicted_kps = model(ims)
        loss = criterion(predicted_kps, kps)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.data.item()
        
    train_loss = train_loss / len(train_loader)
    print(f"Train: loss: {(train_loss * 100):>0.2f}") 
    
    #save_samples(train_loader.dataset, model, f"{script_dir}/results-train.png")
    return train_loss

def test_loop(val_loader, model, criterion, epoch):
    val_loss = 0.0
    model.eval()

    for data in track(val_loader, "Testing... "):
        ims, kps = data

        predicted_kps = model(ims)
        loss = criterion(predicted_kps, kps)

        val_loss += loss.data.item()

    val_loss = val_loss / len(val_loader)
    print(f"Eval: loss: {(val_loss * 100):>0.2f}") 

    save_samples(val_loader.dataset, model, f"{script_dir}/results-{epoch + 1}.png")
    return val_loss


def main():
    tr_loader, te_loader = get_loaders()
    model = get_net()
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    min_test_loss = float("inf")

    for e in range(EPOCHS):
        print(f"\nEpoch {e+1}\n-------------------------------")
        train_loop(tr_loader, model, loss_fn, optimizer)
        test_loss = test_loop(te_loader, model, loss_fn, e)

        if test_loss < min_test_loss:
            min_test_loss = test_loss

            print(f"Saving best model")
            dark.save(model, model_path)

    print("Done!")

if __name__ == "__main__":            
    main()


