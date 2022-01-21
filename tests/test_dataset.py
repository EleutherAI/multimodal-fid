
from itertools import islice
from os.path import dirname, abspath, join
import sys
from torchvision.transforms import Compose, Resize, ToTensor

test_dir = abspath(dirname(__file__))
data_dir = join(test_dir, 'data')
proj_dir = dirname(test_dir)
sys.path.append(test_dir) # fucking cursed

from ttig.dataset import build_webdataset
from ttig.sentence_transformer import build_tokenizer


def test_webdataset():

    def ident(x):
        return x

    im_fn = Compose([Resize(256), ToTensor()])
    
    data = build_webdataset(data_dir, im_fn, ident)
    for image, text in islice(data, 0, 10):
        assert image.shape == (3, 256, 256)
        assert isinstance(text, str)


def test_batched_webdataset():

    im_fn = Compose([Resize(256), ToTensor()])
    data = build_webdataset(
        data_dir,
        im_fn,
        build_tokenizer()
    ).batched(10)
    images, texts = next(iter(data))
    assert images.shape == (10, 3, 256, 256)
    assert len(texts) == 10 # List of dicts
