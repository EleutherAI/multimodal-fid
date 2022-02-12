import kornia.augmentation as K
import torch
from torch import nn
import torchvision.transforms.functional as tf


def to_pil_image(x):
    """Converts from a tensor to a PIL image."""
    if x.ndim == 4:
        assert x.shape[0] == 1
        x = x[0]
    if x.shape[0] == 1:
        x = x[0]
    return tf.to_pil_image((x.clamp(-1, 1) + 1) / 2)
