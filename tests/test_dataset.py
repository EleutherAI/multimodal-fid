
from inspect import isclass
from itertools import islice
from PIL.Image import Image
from os.path import dirname, abspath, join
import sys

test_dir = abspath(dirname(__file__))
data_dir = join(test_dir, 'data')
proj_dir = dirname(test_dir)
sys.path.append(test_dir) # fucking cursed

from ttig.dataset import build_webdataset


def test_webdataset():

    def ident(x):
        return x
    
    data = build_webdataset(data_dir, ident, ident)
    for image, text in islice(data, 0, 10):
        print(image)
        print(text)
        assert isinstance(image, Image)
        assert isinstance(text, str)
