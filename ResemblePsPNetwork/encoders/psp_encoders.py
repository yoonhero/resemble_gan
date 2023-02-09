import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from collections import namedtuple

from stylegan2.model import EqualLinear

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


class Bottleneck(namedtuple("Block", ["in_channel", "depth", "stride"])):
    """A named tuple describing a ResNet Block."""


def get_block(in_channel, depth, num_units, stride=2):
    return [Bottleneck(in_channel, depth, stride)]+[Bottleneck(depth, depth, 1) for i in range(num_units - 1)]

def get_blocks(num_layers):
    blocks = [
			get_block(in_channel=64, depth=64, num_units=3),
			get_block(in_channel=64, depth=128, num_units=4),
			get_block(in_channel=128, depth=256, num_units=14),
			get_block(in_channel=256, depth=512, num_units=3)
	]
    return blocks


class bottleneck_IR(nn.Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = nn.MaxPool2d(1, stride)
        else: 
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channel, depth,(1,1), stride, bias=False),
                nn.BatchNorm2d(depth)
            )
        

class BackboneEncoderUsingLastLayerIntoWPlus(nn.Module):
    def __init__(self, num_layers, mode="ir", opts=None):
        super(BackboneEncoderUsingLastLayerIntoWPlus, self).__init__()

        assert num_layers in [50, 100, 152]
        assert mode in ["ir", 'ir_se']

        ## TODO: Blocks with various number of layers.
        blocks = get_blocks(num_layers)
        unit_module = bottleneck_IR
        # elif mode == "ir_se":
        #     unit_module = bottleneck_IR_SE

        self.n_styles = opts.n_styles
        self.input_layer = nn.Sequential(nn.Conv2d(opts.input_nc, 64, (3, 3), 1, 1, bias=False),
                                         nn.BatchNorm2d(64),
                                         nn.PReLU(64))
        self.output_layer_2 = nn.Sequential(nn.BatchNorm2d(512),
                                            nn.AdaptiveAvgPool2d((7, 7)),
                                            Flatten(),
                                            nn.Linaer(512*7*7, 512))

        self.linear = EqualLinear(512, 512*self.n_styles, lr_mul=1)
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                               bottleneck.depth, bottleneck.stride))
        self.body = nn.Sequential(*modules)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer_2(x)
        x = self.linear(x)
        x = self.view(-1, self.n_styles, 512)
        return x
