import dark
import dark.tensor as dt
from dark.nn import Module, BCEWithLogitsLoss
from dark.optim import Optimizer, Adam
from dark.utils.data import DataLoader

from modelD import Discriminator, get_netD
from modelG import Generator, get_netG
from dataset import get_loader
from util import save_samples
from rich.progress import track
from config import *


def trainStepD_real(netD, criterion, real):
    label = dt.ones((BATCH_SIZE, 1))
    output = netD(real)
    errD_real = criterion(output, label)
    
    return errD_real

def trainStepD_fake(netD, criterion, fake):    
    label = dt.zeros((BATCH_SIZE, 1))
    output = netD(fake.detach())
    errD_fake = criterion(output, label)
    
    return errD_fake

def trainStepG(netD, criterion, fake):
    label = dt.ones((BATCH_SIZE, 1))
    output = netD(fake)
    errG = criterion(output, label)
    
    return errG

def train_loop(netD: Module, netG: Module, 
               dataLoader: DataLoader, criterion, 
               optimizerD: Optimizer, optimizerG: Optimizer):

    netD.train()
    netG.train()

    for i, (real, _) in enumerate(track(dataLoader, "Training")):

        noise = dt.random.randn(BATCH_SIZE, nz, 1, 1)
        fake = netG(noise)

        #train discriminator
        errD_real = trainStepD_real(netD, criterion, real)
        errD_fake = trainStepD_fake(netD, criterion, fake)
        errD = dark.add(errD_real, errD_fake)
        
        optimizerD.zero_grad()
        errD.backward()
        optimizerD.step()

        #train generator
        errG = trainStepG(netD, criterion, fake)
        
        optimizerG.zero_grad()
        errG.backward()
        optimizerG.step()

        if i % 50 == 0:
            print(f"errD: {errD.data.item():>7f}, errG: {errG.data.item():>7f}")
               
def test_loop(netG: Module, noise, epoch):
    netG.eval()  
    
    fake = netG(noise).detach()
    save_samples(fake.data, f"{script_dir}/results-{epoch + 1}.png")
            
            
netG = get_netG()
netD = get_netD()
data_loader = get_loader()

criterion = BCEWithLogitsLoss()
optimizerD = Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

fixed_noise = dt.random.randn(64, nz, 1, 1)
for e in range(EPOCHS):
    print(f"\nEpoch {e+1}\n-------------------------------")

    train_loop(netD, netG, data_loader, criterion, optimizerD, optimizerG)
    test_loop(netG, fixed_noise, e)

    dark.save(netD, modelD_path)
    dark.save(netG, modelG_path)