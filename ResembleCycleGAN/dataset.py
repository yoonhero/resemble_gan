import os
from torch.utils.data import Dataset
import glob
import numpy as np
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt


class ResembleDataset(Dataset):
    def __init__(self, path_human, path_animal, transform=None):
        self.x = glob.glob(path_human)
        self.y = glob.glob(path_animal)
        self.len_x = len(self.x)
        self.len_y = len(self.y)

        self.transform = transform

        self.length_dataset = max(len(self.x), len(self.y))

    def __len__(self):
        return self.length_dataset

    def open_image(self, path):
        return Image.open(path).convert("RGB")

    def __getitem__(self, idx):
        human_img_path = self.x[idx % self.len_x]
        animal_img_path = self.y[idx % self.len_y]

        human_img_path = human_img_path.replace("\\", "/")
        animal_img_path = animal_img_path.replace("\\", "/")

        human_img = self.open_image(human_img_path)
        animal_img = self.open_image(animal_img_path)

        if self.transform:

            human_img = self.transform(human_img)
            animal_img = self.transform(animal_img)

        return human_img, animal_img


if __name__ == "__main__":
    path_human_image = "dataset/after/human/*.jpg"
    path_animal_image = "dataset/after/animal/*.jpg"
    print(glob.glob(path_human_image))
    transform = T.Compose([T.Resize(512, 512), T.ToTensor(),])

    custom_dataset = ResembleDataset(
        path_human_image, path_animal_image, transform=transform)

    x, y = next(iter(custom_dataset))

    plt.imshow(x)

    plt.show()
