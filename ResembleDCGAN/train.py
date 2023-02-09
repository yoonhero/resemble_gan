import os
import numpy as np
import math
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
import torch.nn as nn
import torch.nn.functional as F
import torch
from tqdm import tqdm

import config
from model import Discriminator, Generator, create_latent_z, initialize_weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

nb_epochs = config.NB_EPOCHS
learning_rate = config.LEARNING_RATE
batch_size = config.BATCH_SIZE
latent_dim = config.LATENT_DIM
img_size = config.IMG_SIZE
sample_interval = config.SAMPLE_INTERVAL

transform = transforms.Compose(
    [
        transforms.Resize((64, 64), 0),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)
dataloader = DataLoader(
    datasets.CIFAR10("../dataset/CIFAR10", train=True,
                     download=True, transform=transform),
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
)


generator = Generator(latent_dim, 3, img_size).to(device)
discriminator = Discriminator(3, img_size).to(device)
# initialize_weights(generator)
# initialize_weights(discriminator)

# Optimizers
optimizer_G = torch.optim.Adam(
    generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(
    discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

adversarial_loss = torch.nn.BCELoss()


def save_checkpoint(G, D, epoch):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": D.state_dict(),
            "optimizer_state_dict": optimizer_D.state_dict(),
        },
        "d_model.pt",
    )
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": G.state_dict(),
            "optimizer_state_dict": optimizer_G.state_dict(),
        },
        "g_model.pt",
    )


train_loader = tqdm(dataloader, desc="Loading Train DataLoader")

for epoch in range(nb_epochs):
    G_loss = 0
    D_loss = 0
    for idx, (x, _) in enumerate(train_loader):
        image = x.to(device)

        z = torch.randn(batch_size, latent_dim, 1, 1).to(device)
        fake_image = generator(z)

        disc_real = discriminator(image).reshape(-1)
        real_loss = adversarial_loss(
            disc_real, torch.ones_like(disc_real).to(device))
        disc_fake = discriminator(fake_image).reshape(-1)
        fake_loss = adversarial_loss(
            disc_fake, torch.zeros_like(disc_fake).to(device))
        d_loss = (real_loss + fake_loss) / 2

        optimizer_D.zero_grad()
        d_loss.backward(retain_graph=True)
        optimizer_D.step()

        output = discriminator(fake_image).reshape(-1)
        g_loss = adversarial_loss(output, torch.ones_like(output).to(device))

        optimizer_G.zero_grad()
        g_loss.backward(retain_graph=True)
        optimizer_G.step()

        G_loss += g_loss.item()
        D_loss += d_loss.item()

        train_loader.set_description(
            f"G_Loss: {G_loss / (idx+1)} | D_Loss: {D_loss / (idx+1)}")

        # print(f"Epoch: {epoch} | G_Loss: {g_loss} | D_Loss: {d_loss}")

    save_path = "images/test"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    save_image(fake_image[:25],
               f"{save_path}/result_{epoch}.png", nrow=5, normalize=True)
    # save_checkpoint(generator, discriminator, epoch)
