import random
import torch


def augment_batch(images: torch.Tensor) -> torch.Tensor:
    # 1/3 clean, 1/3 gaussian, 1/3 s&p per image
    out = images.clone()
    N = images.shape[0]
    for i in range(N):
        r = random.random()
        img = out[i]
        if r < 1/3:
            pass  # leave clean
        elif r < 2/3:
            img.add_(torch.randn_like(img) * 0.3)
            img.clamp_(0.0, 1.0)  # keep values valid
        else:
            pmask = torch.rand_like(img) < 0.05  # pepper
            smask = torch.rand_like(img) < 0.05  # salt
            img[pmask] = 0.0
            img[smask] = 1.0
    return out


def apply_gaussian(images: torch.Tensor, sigma: float=0.3) -> torch.Tensor:
    # add gaussian noise to whole batch
    noisy = images + torch.randn_like(images) * sigma
    return noisy.clamp(0.0, 1.0)


def apply_salt_and_pepper(images: torch.Tensor, rate: float=0.05) -> torch.Tensor:
    # randomly set pixels to 0 or 1
    out = images.clone()
    mask = torch.rand_like(out)
    out[mask < rate] = 0.0
    out[mask > (1.0-rate)] = 1.0
    return out
