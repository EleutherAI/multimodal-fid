from os.path import dirname, abspath, join
import sys

test_dir = abspath(dirname(__file__))
data_dir = join(test_dir, 'data')
proj_dir = dirname(test_dir)
sys.path.append(test_dir) # fucking cursed

from ttig.mmfid import make_folder_generator


def test_folder_generator():
    num_batches = 10
    batch_size = 256
    num_samples = batch_size * num_batches
    data_generator = make_folder_generator(data_dir, batch_size, num_samples)
    count = 0
    for images, texts in data_generator:
        assert images.shape == (batch_size, 3, 256, 256)
        assert texts['input_ids'].shape[0] == batch_size
        count += 1
    assert count == num_batches