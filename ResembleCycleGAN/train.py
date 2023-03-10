import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
from torchvision.utils import save_image
import os
from PIL import Image
import argparse

from dataset import ResembleDataset
# from model import Discriminator, Generator
from discriminator import Discriminator
from generator import Generator
from utils import save_checkpoint, load_checkpoint, clear_cache
import config

parser = argparse.ArgumentParser()
parser.add_argument("--is-colab", dest="is_colab",
                    action="store_true", default=False)
parser.add_argument("--load-model", dest="load_model",
                    action="store_true", default=False)
parser.add_argument("--batch-size", dest="batch_size", type=int, default=1)
parser.add_argument("--google-drive-path", dest="google_drive_path", type=str,
                    default="/content/drive/My Drive")
parser.add_argument("--start-epoch", dest="start_epoch", type=int, default=1)
args = parser.parse_args()


clear_cache()

# Set Device for Training
device = torch.device("cuda")

# Hyper Parameters
if not args.is_colab:
    batch_size = config.BATCH_SIZE
    load_model = config.LOAD_MODEL
    checkpoint_gen_h = config.CHECKPOINT_GEN_H
    ckeckpoint_gen_z = config.CHECKPOINT_GEN_Z
    checkpoint_disc_h = config.CHECKPOINT_DISC_H
    checkpoint_disc_z = config.CHECKPOINT_DISC_Z
    path_human_image = config.PATH_HUMAN_IMAGES
    path_animal_image = config.PATH_ANIMAL_IMAGES
    result_images_path = "saved_images"
    start_epoch = config.END_EPOCH+1

else:
    batch_size = args.batch_size
    load_model = args.load_model
    google_drive_path = args.google_drive_path
    checkpoint_gen_h = google_drive_path + "/models/genh.pth.tar"
    ckeckpoint_gen_z = google_drive_path + "/models/genz.pth.tar"
    checkpoint_disc_h = google_drive_path + "/models/critich.pth.tar"
    checkpoint_disc_z = google_drive_path + "/models/criticz.pth.tar"
    path_human_image = google_drive_path + "/dataset/after/human/*"
    path_animal_image = google_drive_path + "/dataset/after/animal/*"
    trained_model_path = google_drive_path + "/models"
    result_images_path = google_drive_path + "/saved_images"
    start_epoch = args.start_epoch

gaussian_noise_rate = config.GAUSSIAN_NOISE_RATE
num_res_blocks = config.NUM_RES_BLOCKS

learning_rate = config.LEARNING_RATE
LAMBDA_IDENTITY = config.LAMBDA_IDENTITY
LAMBDA_CYCLE = config.LAMBDA_CYCLE
nb_epochs = config.NB_EPOCHS

transform = config.TRANSFORM
transforms = config.TRANSFORMS


dataset = ResembleDataset(path_human_image, path_animal_image, transform)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


target_human, target_animal = next(iter(loader))[0:3]
target_human, target_animal = target_human.to(device), target_animal.to(device)

test_image = Image.open("./test_image.JPG")
test_image = transform(test_image)
test_image = test_image.unsqueeze(0).to(device)


def save_result(gen_Z, gen_H):
    fake_animal = gen_H(target_human)
    fake_human = gen_Z(target_animal)
    test_image_preds = gen_H(test_image)

    save_image(torch.cat((target_human * 0.5 + 0.5, fake_human * 0.5 +
               0.5), dim=0), f"{result_images_path}/human_{epoch}.png")
    save_image(torch.cat((target_animal * 0.5 + 0.5, fake_animal *
               0.5 + 0.5), dim=0), f"{result_images_path}/animal_{epoch}.png")
    save_image(torch.cat((test_image*0.5+0.5, test_image_preds *
               0.5+0.5), dim=0), f"{result_images_path}/test_{epoch}.png")


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

        # 1????????? ???????????? ?????? ???
        D_H_real_loss = mse(D_H_real, torch.ones_like(D_H_real))
        # 0????????? ???????????? ?????? ???
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
        identity_animal = gen_Z(animal)
        identity_human = gen_H(human)
        identity_human_loss = l1(identity_human, human)
        identity_animal_loss = l1(identity_animal, animal)

        G_loss = (
            loss_G_Z
            + loss_G_H
            + LAMBDA_CYCLE*cycle_human_loss
            + LAMBDA_CYCLE*cycle_animal_loss
            + LAMBDA_IDENTITY*identity_human_loss
            + LAMBDA_IDENTITY*identity_animal_loss
        )

        g_loss += G_loss.item()

        opt_gen.zero_grad()
        G_loss.backward()
        opt_gen.step()

        loop.set_postfix(d_loss=d_loss/(idx+1), g_loss=g_loss/(idx+1))

    if not os.path.exists(result_images_path):
        os.makedirs(result_images_path)

    save_result(gen_Z, gen_H)


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


if not os.path.exists(trained_model_path):
    os.makedirs(trained_model_path)


if load_model:
    load_checkpoint(checkpoint_gen_h, gen_H, opt_gen, learning_rate)
    load_checkpoint(ckeckpoint_gen_z, gen_Z, opt_gen, learning_rate)
    load_checkpoint(checkpoint_disc_h, disc_H, opt_disc, learning_rate)
    load_checkpoint(checkpoint_disc_z, disc_Z, opt_disc, learning_rate)


for epoch in range(start_epoch, nb_epochs):
    train_loop(disc_H, disc_Z, gen_Z, gen_H, loader, opt_disc,
               opt_gen, L1, mse,  epoch)

    save_checkpoint(gen_H, opt_gen, filename=checkpoint_gen_h)
    save_checkpoint(gen_Z, opt_gen, filename=ckeckpoint_gen_z)
    save_checkpoint(disc_H, opt_disc, filename=checkpoint_disc_h)
    save_checkpoint(disc_Z, opt_disc, filename=checkpoint_disc_z)
