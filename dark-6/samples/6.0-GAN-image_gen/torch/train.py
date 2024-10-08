import torch
from torch.utils.data import DataLoader
from torch.nn import Module, BCEWithLogitsLoss
from torch.optim import Optimizer, Adam

from modelD import Discriminator, get_netD
from modelG import Generator, get_netG
from dataset import get_loader
from util import save_samples
from rich.progress import track
from config import *


def trainStepD_real(netD, criterion, real):
    label = torch.ones((BATCH_SIZE, 1), device=device)
    output = netD(real)
    errD_real = criterion(output, label)
    
    return errD_real

def trainStepD_fake(netD, criterion, fake):    
    label = torch.zeros((BATCH_SIZE, 1), device=device)
    output = netD(fake.detach())
    errD_fake = criterion(output, label)
    
    return errD_fake

def trainStepG(netD, criterion, fake):
    label = torch.ones((BATCH_SIZE, 1), device=device)
    output = netD(fake)
    errG = criterion(output, label)
    
    return errG

def train_loop(netD: Module, netG: Module, 
               dataLoader: DataLoader, criterion, 
               optimizerD: Optimizer, optimizerG: Optimizer):

    netD.train()
    netG.train()

    for i, (real, _) in enumerate(track(dataLoader, "Training")):
        real = real.to(device)

        noise = torch.randn(BATCH_SIZE, nz, 1, 1, device=device)
        fake = netG(noise)

        #train discriminator
        errD_real = trainStepD_real(netD, criterion, real)
        errD_fake = trainStepD_fake(netD, criterion, fake)
        errD = torch.add(errD_real, errD_fake)
        
        optimizerD.zero_grad()
        errD.backward()
        optimizerD.step()

        #train generator
        errG = trainStepG(netD, criterion, fake)
        
        optimizerG.zero_grad()
        errG.backward()
        optimizerG.step()

        if i % 50 == 0:
            print(f"errD: {errD.item():>7f}, errG: {errG.item():>7f}")

def test_loop(netG: Module, noise, epoch):
    netG.eval()  
    
    fake = netG(noise).detach()
    save_samples(fake, f"{script_dir}/samples-{epoch + 1}.png")          
     
            

netG = get_netG()
netD = get_netD()
dataloader = get_loader()

criterion = BCEWithLogitsLoss()
optimizerD = Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

fixed_noise = torch.randn(64, nz, 1, 1, device=device)
for e in range(EPOCHS):
    print(f"\nEpoch {e+1}\n-------------------------------")

    train_loop(netD, netG, dataloader, criterion, optimizerD, optimizerG)
    test_loop(netG, fixed_noise, e)

    #torch.save(netD, modelD_path)
    #torch.save(netG, modelG_path)