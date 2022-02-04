import clip
from collections import namedtuple
from dataclasses import dataclass
import numpy as np
from omegaconf import OmegaConf
from PIL import ImageFile, Image
from taming.models.vqgan import VQModel
import torch
from torch.nn.functional import normalize, one_hot
from torch import nn, optim
from torchvision.transforms import functional as tf
from torchvision import transforms
from tqdm import tqdm
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
from typing import List, Optional, Tuple
from vqgan_clip.masking import MakeCutouts, MakeCutoutsOrig
from vqgan_clip.grad import ReplaceGrad, ClampWithGrad
ImageFile.LOAD_TRUNCATED_IMAGES = True


patch_typeguard()


EmbedTensor = TensorType[-1, 'embedding_dim']
VQCodeTensor = TensorType[-1, 'code_dim', 'x_tokens', 'y_tokens']
ImageTensor = TensorType[-1, 'channels', 'size_x', 'size_y']



@typechecked
def spherical_dist_loss(
    x: TensorType['num_cuts', 'batch', 'embedding_dim'], # One embedding per cut per image in the batch
    y: TensorType[-1, 'batch', 'embedding_dim'] # One embedding for the prompt, extra empty for the number of cuts
) -> TensorType[-1]:
    x = normalize(x, dim=1)
    y = normalize(y, dim=1)
    return (x - y).norm(dim=1).div(2).arcsin().pow(2).mul(2).mean(0)


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


class VqGanClipGenerator(nn.Module):

    def __init__(self, checkpoint_path, model_config_path, config, clip_model_type='', device='cuda'):
        super().__init__()
        self.vqgan = load_vqgan_model(checkpoint_path, model_config_path).to(device)
        self.clip = clip.load(clip_model_type)[0].eval().requires_grad_(False).to(device)
        self.device = device
        self.config = config
        self.normalizer = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
        self.cutout_fn = cutout_factory(
            self.config.cut_method,
            self.clip.visual.input_resolution,
            self.config.num_cuts,
            self.config.cut_pow,
            self.config.augments
        )
    
    @staticmethod
    def replace_grad(x, y):
        return ReplaceGrad.apply(x, y)

    @staticmethod
    def clamp_with_grad(x, y, z):
        return ClampWithGrad.apply(x, y, z)

    @typechecked
    def random_image(
        self,
        rand_im_type: str,
        size: Tuple[int, int],
        batch: int, 
        sides: Tuple[int]
    ) ->  VQCodeTensor:
        if rand_im_type == 'pixels':
            images = [random_noise_image(*size) for _ in range(batch)]
        elif rand_im_type == 'gradient':
            images = [random_gradient_image(*size) for _ in range(batch)]
        else:
            raise ValueError(f'Unknown random initialization strategy {rand_im_type}')
        pil_images = [
            tf.to_tensor(im.resize(sides, Image.LANCZOS)).unsqueeze(0)
            for im in images
        ]
        pil_tensor = torch.concat(pil_images).to(self.device)
        z, *_ = self.vqgan.encode(pil_tensor * 2 - 1)
        return z

    def make_cutouts(self, x: ImageTensor):
        return self.cutout_fn(x)

    @property
    def codebook(self):
        return self.vqgan.quantize.embedding.weight

    def normalize(self, x):
        return self.normalizer(x)

    def vector_quantize(self, x):
        d = x.pow(2).sum(dim=-1, keepdim=True) + self.codebook.pow(2).sum(dim=1) - 2 * x @ self.codebook.T
        indices = d.argmin(-1)
        x_q = one_hot(indices, self.codebook.shape[0]).to(d.dtype) @ self.codebook
        return self.replace_grad(x_q, x)

    def generate_image(self, z: VQCodeTensor) -> ImageTensor:
        z_q = self.vector_quantize(z.movedim(1, 3)).movedim(3, 1)
        return self.clamp_with_grad(self.vqgan.decode(z_q).add(1).div(2), 0, 1)

    @typechecked
    def update_step(self, z: VQCodeTensor, prompts: EmbedTensor) -> TensorType[-1]:
        out = self.generate_image(z)
        image_encodings: EmbedTensor = self.clip.encode_image(
            self.normalize(self.make_cutouts(out))
        ).float()
        dists = spherical_dist_loss(
            image_encodings.view([self.config.num_cuts, out.shape[0], -1]),
            prompts[None]
        )
        return dists # return loss

    @typechecked
    def generate(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        batch_size = len(texts)
        f = 2 ** (self.vqgan.decoder.num_resolutions - 1) # TODO: What number is this usually?
        toksX, toksY = self.config.size[0] // f, self.config.size[1] // f
        sides = (toksX * f, toksY * f)
        # TODO: What is this?????
        e_dim = self.vqgan.quantize.e_dim
        n_toks = self.vqgan.quantize.n_e
        z_min = self.vqgan.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
        z_max = self.vqgan.quantize.embedding.weight.max(dim=0).values[None, :, None, None]   
        if self.config.init_noise in ('pixels', 'gradient'):
            z = self.random_image(self.config.init_noise, self.config.image_size, batch_size, sides)
        else:
            one_hot_embeds = one_hot(torch.randint(n_toks, [toksY * toksX], device=self.device), n_toks).float()        
            z = one_hot_embeds @ self.vqgan.quantize.embedding.weight
            z = z.view([-1, toksY, toksX, e_dim]).permute(0, 3, 1, 2) 
        z.requires_grad_(True)    
        prompts = []

        # CLIP tokenize/encode
        # TODO: Figure out whether this is for multiple-prompts-but-one-image or a batch of images
        
        prompts: EmbedTensor = self.clip.encode_text(clip.tokenize(texts).to(self.device)).float()
        #prompts.append(Prompt(embed).to(self.device))

        # Set the optimiser
        opt = optim.AdamW([z], lr=self.config.step_size)

        for _ in tqdm(range(self.config.max_iterations)):
            # Change text prompt
            opt.zero_grad(set_to_none=True)
            loss = self.update_step(z, prompts)
            loss.backward()
            opt.step()
            
            #with torch.no_grad():
            with torch.inference_mode():
                z.copy_(z.maximum(z_min).minimum(z_max))  # what does this do?
        return self.generate_image(z)





