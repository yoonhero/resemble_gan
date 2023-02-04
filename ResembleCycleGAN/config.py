import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Hyper Parameters
LEARNING_RATE = 0.0002
LAMBDA_IDENTITY = 0.0
LAMBDA_CYCLE = 10

NB_EPOCHS = 1500
END_EPOCH = 13
BATCH_SIZE = 4

GAUSSIAN_NOISE_RATE = 0.05
NUM_RES_BLOCKS = 9

LOAD_MODEL = True
CHECKPOINT_GEN_H = "./models/genh.pth.tar"
CHECKPOINT_GEN_Z = "./models/genz.pth.tar"
CHECKPOINT_DISC_H = "./models/critich.pth.tar"
CHECKPOINT_DISC_Z = "./models/criticz.pth.tar"

PATH_HUMAN_IMAGES = "../dataset/before/human/*"
PATH_ANIMAL_IMAGES = "../dataset/before/animal/*"

TRANSFORM = T.Compose([T.Resize((256, 256), 0), T.ToTensor(),])
TRANSFORMS = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[
                    0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
    ],
    additional_targets={"image0": "image"},
)
