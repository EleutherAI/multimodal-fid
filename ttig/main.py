from os.path import abspath, dirname, join
import sys

proj_dir = abspath(dirname(dirname(__file__)))
sys.path.append(proj_dir) # TODO: Proper packaging so this isn't necessary

from ttig.dataset import CoCa3mTextDataset
from ttig.mmfid import make_reference_statistics, calc_mmfid_from_model, write_images_to_disk
from ttig.models.model import MultiModalFeatureExtractor
from ttig.models.vqgan_clip import VqGanClipGenerator, VQGANConfig
from ttig.utils import to_pil_image
from tqdm import tqdm
import typer


app = typer.Typer()


def generate_and_save_image(model, captions, keys, model_name):
    image_tensors = model.generate(captions)
    image_tensors.to('cpu').detach()
    write_images_to_disk(
        keys,
        [to_pil_image(im) for im in image_tensors],
        model_name
    )



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
