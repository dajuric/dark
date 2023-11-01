import torch
import torch.nn as nn
from torch.nn import Module, BCEWithLogitsLoss
from torch.optim import Optimizer, Adam
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import *
from modelD import Discriminator
from modelG import Generator
from util import save_samples
from rich.progress import track
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
script_dir = os.path.dirname(os.path.realpath(__file__))
nz = 100
batch_size = 64

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.zeros_(m.bias.data)

    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.zeros_(m.bias.data)


def trainStepD_real(netD, criterion, real):
    label = torch.ones((batch_size, 1), device=device)
    output = netD(real)
    errD_real = criterion(output, label)
    
    return errD_real

def trainStepD_fake(netD, criterion, fake):    
    label = torch.zeros((batch_size, 1), device=device)
    output = netD(fake.detach())
    errD_fake = criterion(output, label)
    
    return errD_fake

def trainStepG(netD, criterion, fake):
    label = torch.ones((batch_size, 1), device=device)
    output = netD(fake)
    errG = criterion(output, label)
    
    return errG

def train_loop(netD: Module, netG: Module, 
               dataLoader: DataLoader, criterion, 
               optimizerD: Optimizer, optimizerG: Optimizer):

    for i, (real, _) in enumerate(track(dataLoader, "Training")):
        real = real.to(device)

        noise = torch.randn(batch_size, nz, 1, 1, device=device)
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
            print(f"errD: {errD.data.item():>7f}, errG: {errG.data.item():>7f}")

def test_loop(netG: Module, noise, epoch):
    netG.eval()  
    
    fake = netG(noise).detach()
    save_samples(fake.data, f"{script_dir}/samples-{epoch}.png")          
     
            
tr = Compose([
    Resize((64 ,64)),
    ToTensor(),
    Normalize(0.5, 0.5)
])
dataset = ImageFolder(f"{script_dir}/../db/", transform=tr)
dataLoader = DataLoader(dataset, batch_size, shuffle=True, drop_last=True)

netG = Generator(nz, 64, 3).to(device)
netG.apply(init_weights)

netD = Discriminator(64, 3).to(device)
netD.apply(init_weights)

criterion = BCEWithLogitsLoss()
optimizerD = Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

fixed_noise = torch.randn(64, nz, 1, 1, device=device)
for e in range(5):
    print(f"\nEpoch {e+1}\n-------------------------------")

    train_loop(netD, netG, dataLoader, criterion, optimizerD, optimizerG)
    test_loop(netG, fixed_noise, e)

    torch.save(netD, f"{script_dir}/netD.pt")
    torch.save(netG, f"{script_dir}/netG.pt")