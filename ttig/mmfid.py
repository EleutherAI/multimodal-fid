from cleanfid.fid import frechet_distance, get_batch_features
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm



def feats_to_stats(features):
    """
    Calculates the 
    :features 
    """
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma



def calculate_features_from_generator(
    mumo_model,
    data_generator,
    ref_mu,
    ref_sigma
):
    """
    
    :mumo_model
    :data_generator
    :ref_mu
    :ref_sigma
    :returns 
    """
    device = mumo_model.device
    data_features = []
    for batch in tqdm(data_generator):
        texts, images = batch
        with torch.no_grad():
            data_features.append(
                mumo_model(texts.to(device), images.to(device))
                .detach()
                .cpu()
                .numpy()
            )
    features = np.concatenate(data_features)
    return feats_to_stats(features)


def get_folder_features(folder_fp, model, num_samples, batch_size):
    pass



def make_reference_statistics(name, model, folder, num_workers, num_samples, batch_size):
    stats_folder = os.path.join(os.path.dirname(__file__), "stats")
    os.makedirs(stats_folder, exist_ok=True)
    split, res = "custom", "na"
    outname = f"{name}.npz"
    outf = os.path.join(stats_folder, outname)
    # if the custom stat file already exists
    if os.path.exists(outf):
        msg = f'The statistics file {name} already exists.\n'
        msg += f'Run `rm {os.path.abspath(name)}` to remove it.'
        raise ValueError(msg)
    # get all inception features for folder images
    features = get_folder_features(
        folder,
        model,
        num=num_samples,
        batch_size=batch_size,
        device=device,
        mode=mode,
        description=f"custom stats: {fbname} : "
    )
    mu, sigma = feats_to_stats(features)
    print(f"saving custom FID stats to {outf}")
    np.savez_compressed(outf, mu=mu, sigma=sigma)
