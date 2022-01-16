from cleanfid.fid import frechet_distance, get_batch_features
import numpy as np
import os
import torch
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
from ttig.dataset import TextImageDataset
from typing import Tuple


STATS_FOLDER = os.path.join(os.path.dirname(__file__), "stats")
MmfidStats = Tuple[np.float32, np.ndarray]


def feats_to_stats(features):
    """
    Calculates the mean `mu` and covariance matrix `sigma` of a multimodal dataset.
    :features 
    """
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma





def calculate_features_from_generator(mumo_model, data_generator):
    """
    
    :mumo_model
    :data_generator
    :returns 
    """
    device = mumo_model.device
    data_features = []
    for batch in tqdm(data_generator):
        images, texts = batch
        with torch.no_grad():
            data_features.append(
                mumo_model(texts.to(device), images.to(device))
                .detach()
                .cpu()
                .numpy()
            )
    features = np.concatenate(data_features)
    return feats_to_stats(features)


def make_folder_generator(folder_fp, batch_size):
    dataset = TextImageDataset(
        folder_fp,
        lambda x: x, # TODO: Add image resizing
        lambda x: x
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=RandomSampler(dataset)
    )


def load_reference_statistics(name: str) -> MmfidStats:
    ref_stat_fp = os.path.join(STATS_FOLDER, f'{name}.npz')
    if not os.path.exists(ref_stat_fp):
        raise ValueError(f'No reference statistics for {name}')
    stats = np.load(ref_stat_fp)
    mu, sigma = stats["mu"], stats["sigma"]
    return mu, sigma


def make_reference_statistics(name: str, model, folder_fp: str, num_samples: int, batch_size: int) -> None:
    """

    :name
    :model
    :folder_fp
    :num_samples
    :batch_size
    """
    os.makedirs(STATS_FOLDER, exist_ok=True)
    outname = f"{name}.npz"
    outf = os.path.join(STATS_FOLDER, outname)
    # if the custom stat file already exists
    if os.path.exists(outf):
        msg = f'The statistics file {name} already exists.\n'
        msg += f'Run `rm {os.path.abspath(name)}` to remove it.'
        raise ValueError(msg)
    # get all inception features for folder images
    data_gen = make_folder_generator(folder_fp, num_samples, batch_size)
    features = calculate_features_from_generator(model, data_gen)
    mu, sigma = feats_to_stats(features)
    print(f"saving custom FID stats to {outf}")
    np.savez_compressed(outf, mu=mu, sigma=sigma)
