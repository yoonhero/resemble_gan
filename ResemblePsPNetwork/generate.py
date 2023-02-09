import dnnlib as dnnlib
import legacy as legacy
import torch
import numpy as np
from gen_images import make_transform


def load_stylegan_network(network_pkl, device):
    print(f"Loading networks from {network_pkl}")

    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)["G_ema"].to(device)

    return G


def generate_images(G, device, seed, translate=0, rotate=0, truncation_psi=1, noise_mode="const"):
    label = torch.zeros([1, G.c_dim], device=device)

    z = torch.from_numpy(np.random.RandomState(
        seed).randn(1, G.z_dim)).to(device)

    if hasattr(G.synthesis, "input"):
        m = make_transform(translate, rotate)
        m = np.linalg.inv(m)
        G.synthesis.input.transform.copy_(torch.from_numpy(m))

    img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    return img
