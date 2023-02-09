from glob import glob
from PIL import Image
from pygifmaker.pygifmaker import GifMaker
from glob import glob


def to_gif():
    images = "./images/*.png"
    image_paths = glob(images)

    image_paths = sorted(image_paths, key=lambda x: int(x.split("\\")[-1].split(".")[0]))

    GifMaker.Make("result.gif", image_paths, 3, 0)


if __name__ == "__main__":
    to_gif()
