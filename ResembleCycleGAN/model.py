"""
CycleGAN Discriminator and Generator Model Architecture Implementation Code.

Programmed by Yoonhero.
*   2023-02-02 Initial Coding
"""

import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(Block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride,
                      1, bias=True, padding_mode="reflect"),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super(Discriminator, self).__init__()

        self.initial = nn.Sequential(
            nn.Conv2d(in_channels,
                      features[0],
                      kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )

        self.model = self._make_layer(features)

    def _make_layer(self, features):
        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(Block(in_channels=in_channels, out_channels=feature,
                          stride=1 if feature == features[-1] else 2))

            in_channels = feature

        layers.append(nn.Conv2d(in_channels, 1, kernel_size=4,
                      stride=1, padding=1, padding_mode="reflect"))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.initial(x)

        return torch.sigmoid(self.model(x))


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_act=True, **kwargs):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      padding_mode="reflect", **kwargs)
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if use_act else nn.Identity()
        )

    def forward(self, x):
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, padding=1),
            ConvBlock(channels, channels, use_act=False,
                      kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, img_channels, num_features=64, num_residuals=9):
        super().__init__()

        self.initial = nn.Sequential(
            nn.Conv2d(img_channels, num_features, kernel_size=7,
                      stride=1, padding=3, padding_mode="reflect"),
            nn.ReLU(inplace=True),
        )

        self.down_blocks = nn.ModuleList(
            [
                ConvBlock(num_features, num_features*2,
                          kernel_size=3, stride=2, padding=1),
                ConvBlock(num_features*2, num_features*4,
                          kernel_size=3, stride=2, padding=1),
            ]
        )

        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(num_features*4) for _ in range(num_residuals)]
        )

        self.up_blocks = nn.ModuleList(
            [
                ConvBlock(num_features*4, num_features*2, down=False,
                          kernel_size=3, stride=2, padding=1, output_padding=1),
                ConvBlock(num_features*2, num_features, down=False,
                          kernel_size=3, stride=2, padding=1, output_padding=1)
            ]
        )

        self.last = nn.Conv2d(num_features*1, img_channels,
                              kernel_size=7, stride=1, padding=3)


def test_dicriminator():
    x = torch.randn((1, 3, 256, 256))
    y = torch.randn((1, 3, 256, 256))

    model = Discriminator(in_channels=3)

    prediction = model(x)

    print(model)
    print(prediction.shape)


if __name__ == "__main__":
    test_dicriminator()
