import torch
from PIL import Image
from torchvision.utils import save_image

from generator import Generator
import config


device = torch.device("cuda")


transform = config.TRANSFORM
num_res_blocks = config.NUM_RES_BLOCKS
checkpoint_file = config.CHECKPOINT_GEN_H

model = Generator(num_res_blocks=num_res_blocks)
checkpoint = torch.load(checkpoint_file, map_location=device)

model.load_state_dict(checkpoint["state_dict"])


def inference(image):
    transformed_image = transform(image)

    batched_image = transformed_image.unsqueeze(0).to(device)

    preds = model(batched_image)

    return preds


if __name__ == "__main__":
    image = Image.open("test_image.JPG")

    result = inference(image)

    print(result.shape)

    save_image(result, "result.png")
