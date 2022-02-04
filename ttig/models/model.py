from cleanfid.features import build_feature_extractor
import clip
from PIL import Image
from ttig.models.sentence_transformer import SentenceTransformer
import torch
from torch.nn.functional import normalize, one_hot
from torch import nn, optim
from torchvision.transforms import functional as tf
from torchvision import transforms
from tqdm import tqdm
from typing import Optional, Tuple
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

from ttig.models.vqgan_clip import cutout_factory, load_vqgan_model, random_gradient_image, random_noise_image
from vqgan_clip.grad import ReplaceGrad, ClampWithGrad


patch_typeguard()


EmbedTensor = TensorType[-1, 'embedding_dim']


@typechecked
def spherical_dist_loss(x: EmbedTensor, y: EmbedTensor) -> TensorType[-1]:
    x = normalize(x, dim=1)
    y = normalize(y, dim=1)
    return (x - y).norm(dim=1).div(2).arcsin().pow(2).mul(2)


class MultiModalFeatureExtractor(nn.Module):
    """
    For calculating the multi-modal / conditional FID of a set of images or model
    """
    def __init__(self, image_model_name: str = 'clean', text_model_path: Optional[str] = None):
        super().__init__()
        self.image_model = build_feature_extractor(image_model_name)
        self.text_model = SentenceTransformer(text_model_path)
    
    def forward(self, texts, images):
        image_features = self.image_model(images)
        text_features = self.text_model(texts)
        return torch.concat([image_features, text_features], dim=1)



class VqGanCLIPGenerator(nn.Module):

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

    def random_image(self, rand_im_type: str, size: Tuple[int, int], batch, side_x, side_y):
        if rand_im_type == 'pixels':
            images = [random_noise_image(*size) for _ in range(batch)]
        elif rand_im_type == 'gradient':
            images = [random_gradient_image(*size) for _ in range(batch)]
        else:
            raise ValueError(f'Unknown random initialization strategy {rand_im_type}')
        pil_images = [
            tf.to_tensor(im.resize((side_x, side_y), Image.LANCZOS)).unsqueeze()
            for im in images
        ]
        pil_tensor = torch.concat(pil_images).to(self.device)
        z, *_ = self.vqgan.encode(pil_tensor * 2 - 1)
        return z

    def make_cutouts(self, x):
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

    def generate_image(self, z):
        z_q = self.vector_quantize(z.movedim(1, 3)).movedim(3, 1)
        return self.clamp_with_grad(self.vqgan.decode(z_q).add(1).div(2), 0, 1)

    def update_step(self, z, prompts: EmbedTensor) -> TensorType[-1]:
        out = self.generate_image(z)
        image_encodings = self.clip.encode_image(self.normalize(self.make_cutouts(out))).float() # Encode most recent image
        dists = spherical_dist_loss(image_encodings, prompts)
        return dists # return loss        

    def generate(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        batch_size = len(texts)
        f = 2 ** (self.vqgan.decoder.num_resolutions - 1) # TODO: What number is this usually?
        toksX, toksY = self.config.size[0] // f, self.config.size[1] // f
        sideX, sideY = toksX * f, toksY * f
        # TODO: What is this?????
        e_dim = self.vqgan.quantize.e_dim
        n_toks = self.vqgan.quantize.n_e
        z_min = self.vqgan.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
        z_max = self.vqgan.quantize.embedding.weight.max(dim=0).values[None, :, None, None]   
        if self.config.init_noise in ('pixels', 'gradient'):
            z = self.random_image(self.config.init_noise, self.config.image_size, batch_size, sideX, sideY)
        else:
            one_hot_embeds = one_hot(torch.randint(n_toks, [toksY * toksX], device=self.device), n_toks).float()        
            z = one_hot_embeds @ self.vqgan.quantize.embedding.weight
            z = z.view([-1, toksY, toksX, e_dim]).permute(0, 3, 1, 2) 
        z.requires_grad_(True)    
        prompts = []

        # CLIP tokenize/encode
        # TODO: Figure out whether this is for multiple-prompts-but-one-image or a batch of images
        
        prompts = self.clip.encode_text(clip.tokenize(texts).to(self.device)).float()
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
