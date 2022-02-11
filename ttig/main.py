from os.path import abspath, dirname, join
import sys

proj_dir = abspath(dirname(dirname(__file__)))
sys.path.append(proj_dir) # TODO: Proper packaging so this isn't necessary

from ttig.mmfid import make_reference_statistics, calc_mmfid_from_model
from ttig.models.model import MultiModalFeatureExtractor
from ttig.models.vqgan_clip import VqGanClipGenerator, VQGANConfig
import typer


app = typer.Typer()


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
        num_samples,
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
