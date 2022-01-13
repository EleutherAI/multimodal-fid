from cleanfid.fid import frechet_distance, get_batch_features
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm



def feats_to_stats(features):
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma



def calculate_mmfid_from_generator(
    mumo_model,
    data_generator,
    ref_mu,
    ref_sigma
):
    device = mumo_model.device
    data_features = []
    for batch in tqdm(data_generator):
        texts, images = batch
        data_features.append(
            mumo_model(texts.to(device), images.to(device))
            .detach()
            .cpu()
            .numpy()
        )
    features = np.concatenate(data_features)
    mu, sigma = feats_to_stats(features)
    return frechet_distance(mu, sigma, ref_mu, ref_sigma)

