from cleanfid.resize import make_resizer
from pathlib import Path
from PIL import Image
import pyarrow.dataset as ds
from torch.utils.data import Dataset, IterableDataset
from typing import Any, Callable, Tuple
import webdataset as wds


def build_resizer(size: Tuple[int, int]):
    return make_resizer("PIL", False, "bicubic", size)


def build_webdataset(data_fp: str, image_preprocess_fn: Callable, text_preprocess_fn: Callable):
    data_dir = Path(data_fp)
    assert data_dir.is_dir()
    data = (
        wds.WebDataset([str(file) for file in data_dir.glob('*.tar')])
        .decode('pil')
        .to_tuple('jpg;png', 'txt')
        .map_tuple(image_preprocess_fn, text_preprocess_fn)
    )
    return data


class CoCa3mTextDataset(IterableDataset):

    def __init__(self, folder_fp):
        super().__init__()
        self.data = ds.dataset(folder_fp, format='parquet')
    
    def __iter__(self):
        for batch in self.data.to_batches():
            yield batch








class MuMoFolderDataset(Dataset):
    """ImageDataset is a pytorch Dataset exposing image and text tensors from a folder of image and text"""
    
    def __init__(self, folder: str, preprocess_image_fn: Callable, preprocess_text_fn: Callable):
        super().__init__()
        path = Path(folder)
        text_files = [*path.glob("**/*.txt")]
        text_files = {text_file.stem: text_file for text_file in text_files}
        if len(text_files) == 0:
                raise ValueError(f'No text files found in {folder}')
        image_files = [
            *path.glob("**/*.png"),
            *path.glob("**/*.jpg"),
            *path.glob("**/*.jpeg"),
            *path.glob("**/*.bmp"),
        ]
        image_files = {image_file.stem: image_file for image_file in image_files}
        if len(image_files) == 0:
            raise ValueError(f'No image files found in {folder}')
        keys = None
        self.keys = text_files.keys() & image_files.keys()
        self.text_files = {k: v for k, v in text_files.items() if k in keys}
        self.text_transform = preprocess_text_fn
        self.image_files = {k: v for k, v in image_files.items() if k in keys}
        self.image_transform = preprocess_image_fn

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, ind):
        key = self.keys[ind]
        try:
            image_file = self.image_files[key]
            image_tensor = self.image_transform(Image.open(image_file))            
            text_file = self.text_files[key]
            caption = text_file.read_text()
            text = self.text_transform(caption)
        except (Image.UnidentifiedImageError, OSError, KeyError, Image.DecompressionBombError,):
            print(f"Failed to load image/text {key}. Skipping.")
            return None  # return None to be filtered in the batch collate_fn
        return image_tensor, text
