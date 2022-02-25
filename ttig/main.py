import os
from os.path import abspath, dirname, join
import sys

proj_dir = abspath(dirname(dirname(__file__)))
sys.path.append(proj_dir) # TODO: Proper packaging so this isn't necessary
from CLIP import clip
import re
import torch
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
from ttig.dataset import CoCa3mTextDataset
from ttig.mmfid import make_reference_statistics, calc_mmfid_from_model
from ttig.models.model import MultiModalFeatureExtractor
from ttig.models.vqgan_clip import VqGanClipGenerator, VQGANConfig
from tqdm import tqdm
import typer


app = typer.Typer()


def clean_text(text):
    text = text.strip()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r' ', '_', text)
    return text


def write_images_to_disk(captions, images, model_name):
    image_dir = f'{proj_dir}/images/{model_name}'
    os.makedirs(image_dir, exist_ok=True)
    print(f'captions: {captions}')
    for caption, image in zip(captions, images):
        clean_caption = clean_text(caption)
        image.save(f"{image_dir}/{clean_caption}.png")


def generate_and_save_image(model, captions, text_embs, model_name):
    image_tensors = model(text_embs)
    image_tensors.to('cpu').detach()
    write_images_to_disk(
        captions,
        [to_pil_image(im) for im in image_tensors],
        model_name
    )


@app.command()
def benchmark(batch_size: int = 4):
    model_name = 'vqgan_imagenet_f16_16384'
    checkpoint_path = join(proj_dir, 'checkpoints', 'vqgan', f'{model_name}.ckpt')
    config_path = join(proj_dir, 'checkpoints', 'vqgan', f'{model_name}.yaml')
    config = VQGANConfig()
    vqgan = VqGanClipGenerator(checkpoint_path, config_path, config)
    vqgan.to('cuda:0')
    clip_model = clip.load('ViT-B/16')[0].eval().requires_grad_(False)
    vqgan = torch.nn.DataParallel(vqgan)
    with open(join(proj_dir, 'benchmarks.txt'), mode='r') as bench_file: 
        dataset = [line.strip() for line in bench_file.readlines()]
    data_loader = DataLoader(dataset, batch_size=batch_size)
    for captions in tqdm(data_loader):
        text_embs = clip_model.encode_text(clip.tokenize(captions).to('cuda:0'))
        generate_and_save_image(vqgan, captions, text_embs, 'vqgan_imagenet')


@app.command()
def make_images(data_fp: str, num_samples: int = 524_288, batch_size: int = 4):
    model_name = 'vqgan_imagenet_f16_16384'
    checkpoint_path = join(proj_dir, 'checkpoints', 'vqgan', f'{model_name}.ckpt')
    config_path = join(proj_dir, 'checkpoints', 'vqgan', f'{model_name}.yaml')
    config = VQGANConfig()
    vqgan = VqGanClipGenerator(checkpoint_path, config_path, config)
    vqgan.to('cuda')
    dataset = CoCa3mTextDataset(data_fp, batch_size=batch_size)
    count = 0
    for keys, captions in tqdm(dataset):
        captions = list(captions)
        count += len(keys)
        if num_samples is not None and count > num_samples:
            break
        # prompts = tokenizer(captions, padding='longest', truncation=True, return_tensors='pt')
        generate_and_save_image(vqgan, captions, keys, 'vqgan_imagenet')
        
        
@app.command()
def mmfid(data_fp: str, ref_stats_name='coco3m_total', num_samples: int = 524_288, batch_size: int = 128):
    model_name = 'vqgan_imagenet_f16_16384'
    checkpoint_path = join(proj_dir, 'checkpoints', 'vqgan', f'{model_name}.ckpt')
    config_path = join(proj_dir, 'checkpoints', 'vqgan', f'{model_name}.yaml')
    config = VQGANConfig()
    stats_model = MultiModalFeatureExtractor()
    stats_model.to('cuda')
    vqgan = VqGanClipGenerator(checkpoint_path, config_path, config)
    vqgan.to('cuda')
    calc_mmfid_from_model(
        data_fp,
        vqgan,
        stats_model,
        ref_stats_name,
        'vqgan_imagenet',
        batch_size,
        num_samples = num_samples,
        save_images = True
    )


@app.command()
def calc_stats(name: str, folder_fp: str, num_samples: int = 500_000, batch_size: int = 128):
    model = MultiModalFeatureExtractor()
    model.to('cuda')
    return make_reference_statistics(
        name,
        model,
        folder_fp,
        num_samples,
        batch_size
    )


if __name__ == '__main__':
    app()
