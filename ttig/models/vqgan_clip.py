from omegaconf import OmegaConf
from taming.models.vqgan import VQModel


def load_vqgan_model(checkpoint_path: str, config_path: str):
    config = OmegaConf.load(config_path)
    model = VQModel(**config.model.params)
    model.eval().requires_grad_(False)
    model.init_from_ckpt(checkpoint_path)
    return model
