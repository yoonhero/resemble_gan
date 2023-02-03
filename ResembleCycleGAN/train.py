import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
from torchvision.utils import save_image
import numpy as np
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import torch
import gc

from dataset import ResembleDataset
# from model import Discriminator, Generator
from discriminator import Discriminator
from generator import Generator
from utils import save_checkpoint

# Clean Cache
gc.collect()
torch.cuda.empty_cache()

# Set Device for Training
device = torch.device("cuda" if torch.cuda.is_available else "cpu")

# Hyper Parameters
learning_rate = 0.0002
LAMBDA_IDENTITY = 0.0
LAMBDA_CYCLE = 10
nb_epochs = 100
gaussian_noise_rate = 0.05
num_res_blocks = 9


# Train function for 1 Epoch.
def train_loop(disc_H, disc_Z, gen_Z, gen_H, loader, opt_disc, opt_gen, l1, mse, epoch):
    d_loss = 0
    g_loss = 0

    loop = tqdm(loader, leave=True)

    disc_H.train()
    disc_Z.train()
    gen_H.train()
    gen_Z.train()

    for idx, (human, animal) in enumerate(loop):
        human, animal = human.to(device), animal.to(device)

        ## Discriminator ##
        fake_animal = gen_H(human)

        D_H_real = disc_H(animal.detach())
        D_H_fake = disc_H(fake_animal.detach())

        # 1이라고 판별하면 옳은 거
        D_H_real_loss = mse(D_H_real, torch.ones_like(D_H_real))
        # 0이라고 판별해야 옳은 거
        D_H_fake_loss = mse(D_H_fake, torch.zeros_like(D_H_fake))
        D_H_loss = D_H_real_loss + D_H_fake_loss

        fake_human = gen_Z(animal)

        D_Z_real = disc_Z(human.detach())
        D_Z_fake = disc_Z(fake_human.detach())

        D_Z_real_loss = mse(D_Z_real, torch.ones_like(D_Z_real))
        D_Z_fake_loss = mse(D_Z_fake, torch.zeros_like(D_Z_fake))
        D_Z_loss = D_Z_real_loss + D_Z_fake_loss

        D_loss = (D_H_loss + D_Z_loss) / 2

        d_loss += D_loss.item()

        opt_disc.zero_grad()
        D_loss.backward(retain_graph=True)
        opt_disc.step()

        ## GENERATORS ##

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

        g_loss += G_loss.item()

        opt_gen.zero_grad()
        G_loss.backward()
        opt_gen.step()

        loop.set_postfix(d_loss=d_loss/(idx+1), g_loss=g_loss/(idx+1))

    if not os.path.exists("saved_images/"):
        os.makedirs("saved_images")
    save_image(torch.concat((fake_human * 0.5 + 0.5, human * 0.5 +
               0.5), dim=0), f"saved_images/human_{epoch}.png")
    save_image(torch.concat((fake_animal * 0.5 + 0.5, animal *
               0.5 + 0.5), dim=0), f"saved_images/animal_{epoch}.png")


disc_H = Discriminator(gaussian_noise_rate=gaussian_noise_rate).to(device)
disc_Z = Discriminator(gaussian_noise_rate=gaussian_noise_rate).to(device)
gen_Z = Generator(num_res_blocks=num_res_blocks).to(device)
gen_H = Generator(num_res_blocks=num_res_blocks).to(device)

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
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[
                    0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
    ],
    additional_targets={"image0": "image"},
)


dataset = ResembleDataset(path_human_image, path_animal_image, transform)
loader = DataLoader(dataset, batch_size=1, shuffle=True)

g_scaler = torch.cuda.amp.GradScaler()
d_scaler = torch.cuda.amp.GradScaler()


if not os.path.exists("./models/"):
    os.makedirs("./models/")


for epoch in range(nb_epochs):
    train_loop(disc_H, disc_Z, gen_Z, gen_H, loader, opt_disc,
               opt_gen, L1, mse,  epoch)

    save_checkpoint(gen_H, opt_gen, filename="./models/genh.pth.tar")
    save_checkpoint(gen_Z, opt_gen, filename="./models/genz.pth.tar")
    save_checkpoint(disc_H, opt_disc, filename="./models/critich.pth.tar")
    save_checkpoint(disc_Z, opt_disc, filename="./models/criticz.pth.tar")
