from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image


class TextImageDataset(Dataset):
    """ImageDataset is a pytorch Dataset exposing image and text tensors from a folder of image and text"""
    
    def __init__(self, folder, preprocess_image_fn, preprocess_text_fn):
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
