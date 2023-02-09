import torch
import torch.nn as nn
import torch.nn.functional as F


def create_latent_z(batch_size, latent_dim):
    return torch.randn((batch_size, latent_dim))*(0.1**0.5)


class Generator(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g):
        super(Generator, self).__init__()
#         self.l1 = nn.Sequential(nn.Linear(latent_dim, 128*self.init_size**2))

#         self.conv_blocks = nn.Sequential(
#             nn.BatchNorm2d(128),
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(128, 128, 3, stride=1, padding=1),
#             nn.BatchNorm2d(128, 0.8),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(128, 64, 3, stride=1, padding=1),
#             nn.BatchNorm2d(64, 0.8),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(64, 3, 3, stride=1, padding=1),
#             nn.Tanh()
#         )
        self.net = nn.Sequential(
            self._block(channels_noise, features_g*16, 4, 1, 0),  # img = 4*4
            self._block(features_g * 16, features_g * 8, 4, 2, 1),
            self._block(features_g * 8, features_g*4, 4, 2, 1),  # img = 8*8
            self._block(features_g * 4, features_g*2, 4, 2, 1),  # img: 16 * 16
            nn.ConvTranspose2d(
                features_g * 2, channels_img, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=True,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        # out = self.l1(x)
        # out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        # img = self.conv_blocks(out)
        return self.net(x)


class Discriminator(nn.Module):
    """
    64*64 => 66*66
    kernel: 3 | stride: 2 | padding: 1 => 32*32
    kernel: 3 | stride: 2 | padding: 1 => 16*16
    ...
    output size = ((input size) + 2 * padding - (kernel size - 1))/stride+1 내림
    """

    def __init__(self, channels_img, features_d):
        super(Discriminator, self).__init__()
        # self.channels = channels
        # self.layers = self._make_layer()

        self.disc = nn.Sequential(
            nn.Conv2d(channels_img, features_d, 4, 2, 1),
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d * 2, 4, 2, 1),
            self._block(features_d * 2, features_d * 4, 4, 2, 1),
            self._block(features_d * 4, features_d * 8, 4, 2, 1),
            nn.Conv2d(features_d * 8, 1, 4, 2, 0),
            nn.Sigmoid(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
        # ds_size = img_size // 2 ** 4
        # self.adv_layer = nn.Sequential(
        #     nn.Linear(128*ds_size**2, 1), nn.Sigmoid())

    # def _make_layer(self):
    #     layers = []
    #     assert self.channels != []

    #     def discriminator_block(in_filters, out_filters, bn=True):
    #         block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1),
    #                  nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
    #         if bn:
    #             block.append(nn.BatchNorm2d(out_filters, 0.8))

    #         return block

    #     for idx, channel in enumerate(self.channels[:-1]):
    #         block = discriminator_block(channel, self.channels[idx+1])
    #         for b in block:
    #             layers.append(b)

    #     return nn.Sequential(*layers)

    def forward(self, x):
        # x = self.layers(x)
        # x = x.view(x.shape[0], -1)
        # validity = self.adv_layer(x)
        return self.disc(x)


def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


def test():
    N, in_channels, H, W = 8, 3, 64, 64
    noise_dim = 100
    x = torch.randn((N, in_channels, H, W))
    disc = Discriminator(in_channels, 8)
    assert disc(x).shape == (N, 1, 1, 1)
    gen = Generator(noise_dim, in_channels, 8)
    z = torch.randn((N, noise_dim, 1, 1))
    assert gen(z).shape == (N, in_channels, H, W)
    print("Success")


if __name__ == '__main__':
    # z = create_latent_z()
    test()
