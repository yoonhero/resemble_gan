import torch
import torch.optim as optim
from torch.utils.data import DataLoader 
import torch.nn as nn
from tqdm import tqdm 
from torchvision.utils import save_image
from torch.utils.data import Dataset
import glob
import numpy as np
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import torch, gc

from dataset import ResembleDataset
from model import Discriminator, Generator
from utils import save_checkpoint

## Clean Cache
gc.collect()
torch.cuda.empty_cache()

## Set Device for Training
device = torch.device("cuda" if torch.cuda.is_available else "cpu")

## Hyper Parameters
learning_rate = 1e-5
LAMBDA_IDENTITY = 0.0
LAMBDA_CYCLE = 10
nb_epochs = 100



## Train function for 1 Epoch.
def train_loop(disc_H, disc_Z,gen_Z, gen_H, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler, epoch):
    l_reals = 0
    l_fakes = 0

    loop = tqdm(loader, leave=True)

    disc_H.train()
    disc_Z.train()
    gen_H.train()
    gen_Z.train()

    for idx, (human, animal) in enumerate(loop):
        human, animal = human.to(device), animal.to(device)
        
        with torch.cuda.amp.autocast():
            fake_animal = gen_H(human)

            D_H_real = disc_H(animal)
            D_H_fake = disc_H(fake_animal.detach())

            l_reals += D_H_real.mean().item()
            l_fakes += D_H_fake.mean().item()

            # 1이라고 판별하면 옳은 거 
            D_H_real_loss = mse(D_H_real, torch.ones_like(D_H_real))
            # 0이라고 판별해야 옳은 거 
            D_H_fake_loss = mse(D_H_fake, torch.zeros_like(D_H_fake))

            D_H_loss = D_H_real_loss + D_H_fake_loss


            fake_human = gen_Z(animal)

            D_Z_real = disc_Z(human)
            D_Z_fake = disc_Z(fake_human.detach())

            D_Z_real_loss = mse(D_Z_real, torch.ones_like(D_Z_real))
            D_Z_fake_loss = mse(D_Z_fake, torch.zeros_like(D_Z_fake))

            D_Z_loss = D_Z_real_loss + D_Z_fake_loss

            D_loss = (D_H_loss + D_Z_loss) / 2
        
        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward(retain_graph=True)
        d_scaler.step(opt_disc)
        d_scaler.update()


        with torch.cuda.amp.autocast():
            D_H_fake = disc_H(fake_animal)
            D_Z_fake = disc_Z(fake_human)
            loss_G_H = mse(D_H_fake, torch.ones_like(D_H_fake))
            loss_G_Z = mse(D_Z_fake, torch.ones_like(D_Z_fake))

            # cycle loss
            cycle_human = gen_Z(fake_animal)
            cycle_animal = gen_H(fake_human)
            cycle_human_loss = l1(human, cycle_human)
            cycle_animal_loss = l1(animal, cycle_animal)

            # identity loss => keep color theme
            # identity_animal = gen_Z(animal)
            # identity_human = gen_H(human)
            # identity_human_loss = l1(identity_human, human)
            # identity_animal_loss = l1(identity_animal, animal)

            G_loss = (
                loss_G_Z 
                + loss_G_H
                 + LAMBDA_CYCLE*cycle_human_loss
                  + LAMBDA_CYCLE*cycle_animal_loss 
                #   + LAMBDA_IDENTITY*identity_human_loss 
                #   + LAMBDA_IDENTITY*identity_animal_loss
            )

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward(retain_graph=True)
        g_scaler.step(opt_gen)
        g_scaler.update()

        loop.set_postfix(l_reals=l_reals/(idx+1), l_fakes=l_fakes/(idx+1))

    if not os.path.exists("saved_images/"):
        os.makedirs("saved_images")
    save_image(fake_human * 0.5 + 0.5, f"saved_images/human_{epoch}.png")
    save_image(fake_animal*0.5 + 0.5, f"saved_images/animal_{epoch}.png")

        

disc_H = Discriminator(in_channels=3).to(config.DEVICE)
disc_Z = Discriminator(in_channels=3).to(config.DEVICE)
gen_Z = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
gen_H = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
opt_disc = optim.Adam(
        list(disc_H.parameters()) + list(disc_Z.parameters()),
        lr=learning_rate,
        betas=(0.5, 0.999),
)

opt_gen = optim.Adam(
        list(gen_Z.parameters()) + list(gen_H.parameters()),
        lr=learning_rate,
        betas=(0.5, 0.999),
)

L1 = nn.L1Loss()
mse = nn.MSELoss()

path_human_image = "../dataset/before/human/*"
path_animal_image = "../dataset/before/animal/*"
    
transform = T.Compose([T.Resize((256, 256), 0), T.ToTensor(),])
transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
    ],
    additional_targets={"image0": "image"},
)


dataset = ResembleDataset(path_human_image, path_animal_image, transform)
loader = DataLoader(dataset, batch_size=1, shuffle=True)

g_scaler = torch.cuda.amp.GradScaler()
d_scaler = torch.cuda.amp.GradScaler()


if os.path.exists("models/"):
    os.makedirs("models/")


for epoch in range(nb_epochs):
    train_loop(disc_H, disc_Z,gen_Z, gen_H, loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler, epoch)

    save_checkpoint(gen_H, opt_gen, filename="./models/genh.pth.tar")
    save_checkpoint(gen_Z, opt_gen, filename="./models/genz.pth.tar")
    save_checkpoint(disc_H, opt_disc, filename="./models/critich.pth.tar")
    save_checkpoint(disc_Z, opt_disc, filename="./models/criticz.pth.tar")