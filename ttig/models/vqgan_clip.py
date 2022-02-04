from collections import namedtuple
from dataclasses import dataclass
from omegaconf import OmegaConf
from taming.models.vqgan import VQModel
from typing import List, Optional, Tuple
from vqgan_clip.masking import MakeCutouts, MakeCutoutsOrig


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





