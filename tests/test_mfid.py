from os.path import dirname, abspath, join
from random import sample
import numpy as np
import sys
import warnings

test_dir = abspath(dirname(__file__))
data_dir = join(test_dir, 'data')
proj_dir = dirname(test_dir)
sys.path.append(test_dir) # fucking cursed
warnings.filterwarnings("ignore", category=DeprecationWarning) 

from ttig.mfid import symmetric_matrix_square_root, mfid


def test_symmetric_matrix_square_root():
    rng = np.random.default_rng()
    x = rng.standard_normal(size=(5, 5))
    x = (x + x.T) / 2 
    x = x + np.diag(np.asarray([1]*5)) * 4 # x needs to be symmetric _and_ positive-semi-definite
    x_squared = x @ x
    x_hat = symmetric_matrix_square_root(x_squared)
    assert np.allclose(x, x_hat), print(x, '\n=========\n', x_hat)


def test_multimodal():
    rng = np.random.default_rng()
    true_cov = [
        [2, 0, 0, 0, 0.5, 0.2],
        [0, 2, 0, 0, 0.5, 0.2],
        [0, 0, 2, 0, 0.5, 0.2],
        [0, 0, 0, 2, 0.5, 0.2],
        [0.5, 0.5, 0.5, 0.5, 1, 0.1],
        [0.2, 0.2, 0.2, 0.2, 0.1, 1]
    ]
    y1_x = rng.multivariate_normal(mean=[0, 0, 0, 0, 3, 3], cov=true_cov, size=(5000,))
    y1, x = y1_x[:, :-2], y1_x[:, -2:]
    assert x.shape == (5000, 2)
    assert y1.shape == (5000, 4)
    y2 = rng.standard_normal(size=(5000, 4)) * 2 # mean=0, var=2 uncorrelated normal, no dependence on x
    assert np.allclose(mfid(y1, y1, x), 0)
    sample_mfid_0 = mfid(y1, y2, x)
    assert sample_mfid_0 > 0 # Probably better to calculate it properly by hand...
    # MFID is invariant to scalar multiplication of x
    assert np.allclose(sample_mfid_0, mfid(y1, y2, x * 3))
