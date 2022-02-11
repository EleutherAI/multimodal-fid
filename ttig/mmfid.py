from cleanfid.fid import frechet_distance
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from ttig.dataset import build_resizer, build_webdataset, CoCa3mTextDataset
from ttig.models.sentence_transformer import build_tokenizer, encoding_to_cuda
from ttig.utils import to_pil_image
from torchvision.transforms import ToTensor, Compose
from typing import Optional, Tuple


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


def calc_mmfid_from_model(
    folder_fp: str,
    gen_model,
    stats_model,
    reference_stats_name: str,
    model_name: str,
    batch_size: int,
    tokenizer = None,
    num_samples: Optional[int] = 524_288, # 2 ** 19
    save_images: bool = True
):
    image_size: Tuple[int, int] = (299, 299)
    os.makedirs(STATS_FOLDER, exist_ok=True)
    outname = f"{model_name}.npz"
    outf = os.path.join(STATS_FOLDER, outname)
    ref_mu, ref_sigma = load_reference_statistics(reference_stats_name) 
    data_gen = make_model_generator(
        folder_fp,
        gen_model,
        model_name,
        batch_size,
        image_size,
        tokenizer,
        num_samples,
        save_images
    )
    features = calculate_features_from_generator(stats_model, data_gen)
    mu, sigma = feats_to_stats(features)
    print(f"Saving custom Multi-Modal FID (MMFID) stats to {outf}")
    np.savez_compressed(outf, mu=mu, sigma=sigma)
    mmfid = frechet_distance(mu, sigma, ref_mu, ref_sigma)
    print(mmfid)
    return mmfid



    


def calculate_features_from_generator(mumo_model, data_generator):
    """
    
    :mumo_model
    :data_generator
    :returns 
    """
    data_features = []
    for batch in tqdm(data_generator):
        images, texts = batch
        with torch.no_grad():
            data_features.append(
                mumo_model(
                    encoding_to_cuda(texts),
                    images.to('cuda')
                )
                .detach()
                .cpu()
                .numpy()
            )
    return np.concatenate(data_features)


def make_folder_generator(folder_fp, batch_size, num_samples: Optional[int] = None, image_size=(299, 299), tokenizer=None):
    # For some reason their pipeline involves loading data as an np.ndarray, converting to an image, and converting back
    # TODO: there has gotta be a better way to do that, but for now I wanna rely on their implementation being correct
    image_fn = Compose([np.asarray, build_resizer(image_size), ToTensor()])
    tokenizer = tokenizer if tokenizer is not None else build_tokenizer()
    def ident(x):
        return x
    dataset = build_webdataset(
        folder_fp,
        image_fn,
        ident
    )
    dataset = dataset.shuffle(1000)
    if num_samples is not None:
        dataset = dataset.slice(num_samples)
    dataset = dataset.batched(batch_size).map_tuple(
        ident, 
        lambda sents: tokenizer(sents, padding='longest', truncation=True, return_tensors='pt')
    )
    return DataLoader(
        dataset,
        batch_size=None,
        pin_memory=True
    )


def write_images_to_disk(keys, images, model_name):
    print('hello I am saving an image')
    image_dir = f'./images/{model_name}'
    os.makedirs(image_dir, exist_ok=True)
    for key, image in zip(keys, images):
        to_pil_image(image).save(f'{image_dir}/{key}.png')


def make_model_generator(
    folder_fp: str,
    model,
    model_name: str,
    batch_size: int,
    image_size: Tuple[int, int] = (299, 299),
    tokenizer = None,
    num_samples: Optional[int] = None,
    save_images: bool = True
):
    # For some reason their pipeline involves loading data as an np.ndarray, converting to an image, and converting back
    # TODO: there has gotta be a better way to do that, but for now I wanna rely on their implementation being correct
    image_fn = Compose([np.asarray, build_resizer(image_size), ToTensor()])
    tokenizer = tokenizer if tokenizer is not None else build_tokenizer()
    dataset = CoCa3mTextDataset(folder_fp, batch_size=batch_size)
    count = 0
    with ThreadPoolExecutor(max_workers=32) as executor:
        for keys, captions in dataset:
            captions = list(captions)
            count += len(keys)
            if num_samples is not None and count > num_samples:
                break
            prompts = tokenizer(captions, padding='longest', truncation=True, return_tensors='pt')
            images = model.generate(prompts)
            if save_images:
                executor.submit(write_images_to_disk, keys, images, model_name)
            yield prompts, image_fn(images.to('cpu').detach())


def load_reference_statistics(name: str) -> MmfidStats:
    ref_stat_fp = os.path.join(STATS_FOLDER, f'{name}.npz')
    if not os.path.exists(ref_stat_fp):
        raise ValueError(f'No reference statistics for {name}')
    stats = np.load(ref_stat_fp)
    mu, sigma = stats["mu"], stats["sigma"]
    return mu, sigma


def make_reference_statistics(name: str, model, folder_fp: str, num_samples: int, batch_size: int) -> None:
    """

    :name str  Name of the 
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
    data_gen = make_folder_generator(folder_fp, batch_size, num_samples)
    features = calculate_features_from_generator(model, data_gen)
    mu, sigma = feats_to_stats(features)
    print(f"Saving custom Multi-Modal FID (MMFID) stats to {outf}")
    np.savez_compressed(outf, mu=mu, sigma=sigma)
