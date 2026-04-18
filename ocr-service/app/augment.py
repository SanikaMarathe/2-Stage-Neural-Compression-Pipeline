"""
Noise augmentation utilities for digit recognition training and evaluation.
"""

import random

import torch


def augment_batch(images: torch.Tensor) -> torch.Tensor:
    """Per-image random: 1/3 clean, 1/3 Gaussian(sigma=0.3 clamped [0,1]), 1/3 salt-and-pepper(5%).
    Input/output shape: (N, 1, 28, 28), float32 in [0,1]. Operates in-place on a clone."""
    out = images.clone()
    N = images.shape[0]

    for i in range(N):
        r = random.random()
        img = out[i]  # view into out, shape (1, 28, 28)

        if r < 1 / 3:
            # clean — no modification
            pass
        elif r < 2 / 3:
            # Gaussian noise
            img.add_(torch.randn_like(img) * 0.3)
            img.clamp_(0.0, 1.0)
        else:
            # Salt-and-pepper
            pepper_mask = torch.rand_like(img) < 0.05
            salt_mask = torch.rand_like(img) < 0.05
            img[pepper_mask] = 0.0
            img[salt_mask] = 1.0

    return out


def apply_gaussian(images: torch.Tensor, sigma: float = 0.3) -> torch.Tensor:
    """
    Apply Gaussian noise to the entire batch.

    Args:
        images: Tensor of shape (N, 1, 28, 28), values in [0, 1].
        sigma:  Standard deviation of the Gaussian noise.

    Returns:
        Noisy tensor, values clamped to [0, 1].
    """
    noisy = images + torch.randn_like(images) * sigma
    return noisy.clamp(0.0, 1.0)


def apply_salt_and_pepper(images: torch.Tensor, rate: float = 0.05) -> torch.Tensor:
    """
    Apply salt-and-pepper noise to the entire batch.

    Pixels with mask < rate  → 0 (pepper)
    Pixels with mask > 1-rate → 1 (salt)

    Args:
        images: Tensor of shape (N, 1, 28, 28), values in [0, 1].
        rate:   Fraction of pixels set to 0 (pepper) and 1 (salt) respectively.

    Returns:
        Corrupted tensor with values in [0, 1].
    """
    out = images.clone()
    mask = torch.rand_like(out)
    out[mask < rate] = 0.0
    out[mask > (1.0 - rate)] = 1.0
    return out
