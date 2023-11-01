import dark
import dark.tensor as dt
import dark.nn as nn
from dark.nn import Module, BCEWithLogitsLoss
from dark.optim import Optimizer, Adam
from dark.utils.data import ImageFolder, DataLoader
from dark.utils.transforms import *
from modelD import Discriminator
from modelG import Generator
from util import save_samples
import pickle
from rich.progress import track
import os

print(f"Running on: {'cuda' if dt.is_cuda() else 'cpu'}")
script_dir = os.path.dirname(os.path.realpath(__file__))
nz = 100
batch_size = 64

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        m.weights.data = dt.random.normal(0.0, 0.02, m.weights.data.shape)
        m.bias.data = dt.zeros_like(m.bias.data)

    if isinstance(m, nn.Linear):
        m.weights.data = dt.random.normal(1.0, 0.02, m.weights.data.shape)
        m.bias.data = dt.zeros_like(m.bias.data)


def trainStepD_real(netD, criterion, real):
    label = dt.ones((batch_size, 1))
    output = netD(real)
    errD_real = criterion(output, label)
    
    return errD_real

def trainStepD_fake(netD, criterion, fake):    
    label = dt.zeros((batch_size, 1))
    output = netD(fake.detach())
    errD_fake = criterion(output, label)
    
    return errD_fake

def trainStepG(netD, criterion, fake):
    label = dt.ones((batch_size, 1))
    output = netD(fake)
    errG = criterion(output, label)
    
    return errG

def train_loop(netD: Module, netG: Module, 
               dataLoader: DataLoader, criterion, 
               optimizerD: Optimizer, optimizerG: Optimizer):

    netD.train()
    netG.train()

    for i, (real, _) in enumerate(track(dataLoader, "Training")):

        noise = dt.random.randn(batch_size, nz, 1, 1)
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
    save_samples(fake.data, f"{script_dir}/samples-{epoch}.png")
            
            
tr = Compose(
    Resize(64 ,64),
    Normalize(0.5, 0.5),
    ToTensorV2(),
)
dataset = ImageFolder(f"{script_dir}/../db/", imgT=tr)
dataLoader = DataLoader(dataset, batch_size, shuffle=True, drop_last=True)

netG = Generator(nz, 64, 3)
netG.apply(init_weights)

netD = Discriminator(64, 3)
netD.apply(init_weights)

criterion = BCEWithLogitsLoss()
optimizerD = Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

fixed_noise = dt.random.randn(64, nz, 1, 1)
for e in range(1):
    print(f"\nEpoch {e+1}\n-------------------------------")

    train_loop(netD, netG, dataLoader, criterion, optimizerD, optimizerG)
    test_loop(netG, fixed_noise, e)

    pickle.dump(netD, open(f"{script_dir}/netD.pt", "wb"))
    pickle.dump(netG, open(f"{script_dir}/netG.pt", "wb"))