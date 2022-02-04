from collections import namedtuple
from dataclasses import dataclass
from omegaconf import OmegaConf
from taming.models.vqgan import VQModel
from typing import List, Optional, Tuple
from vqgan_clip.masking import MakeCutouts, MakeCutoutsOrig

import numpy as np
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True

def random_noise_image(w,h):
    random_image = Image.fromarray(np.random.randint(0,255,(w,h,3),dtype=np.dtype('uint8')))
    return random_image


# create initial gradient image
def gradient_2d(start, stop, width, height, is_horizontal):
    if is_horizontal:
        return np.tile(np.linspace(start, stop, width), (height, 1))
    else:
        return np.tile(np.linspace(start, stop, height), (width, 1)).T


def gradient_3d(width, height, start_list, stop_list, is_horizontal_list):
    result = np.zeros((height, width, len(start_list)), dtype=float)

    for i, (start, stop, is_horizontal) in enumerate(zip(start_list, stop_list, is_horizontal_list)):
        result[:, :, i] = gradient_2d(start, stop, width, height, is_horizontal)

    return result

    
def random_gradient_image(w,h):
    array = gradient_3d(w, h, (0, 0, np.random.randint(0,255)), (np.random.randint(1,255), np.random.randint(2,255), np.random.randint(3,128)), (True, False, False))
    random_image = Image.fromarray(np.uint8(array))
    return random_image


@dataclass
class VQGANConfig:
    num_cuts: int = 32
    num_iterations: int = 500
    cut_method: str = 'latest' # other option is 'original'
    cut_pow: float = 1.0
    learning_rate: float = 0.1
    init_noise: Optional[str] = None
    augments: Optional[List] = None
    size: Tuple[int] = (256, 256)
    step_size: float = 0.1
    max_iterations: int = 500

    
    def __post_init__(self):
        if self.augments is None:
            # TODO: Make this more readable
            self.augments = ['Hf','Af', 'Pe', 'Ji', 'Er'] # *sigh*


def load_vqgan_model(checkpoint_path: str, config_path: str):
    config = OmegaConf.load(config_path)
    model = VQModel(**config.model.params)
    model.eval().requires_grad_(False)
    model.init_from_ckpt(checkpoint_path)
    return model


CutoutConfig = namedtuple('CutoutConfig', ['cut_pow', 'augments']) # Fucking cursed 
# TODO: refactor vqgan-clip code so that their cutout making class isn't dependent on their command line args....


def cutout_factory(cut_method: str, cut_size, num_cuts: int, cut_pow: float, augments: List[str]):
    not_args = CutoutConfig(cut_pow, augments)
    if cut_method == 'original':
        return MakeCutoutsOrig(not_args, cut_size, num_cuts)
    elif cut_method == 'latest':
        return MakeCutouts(not_args, cut_size, num_cuts)
    else:
        raise ValueError(f'Not recognized cutout-making type {cut_method}')





