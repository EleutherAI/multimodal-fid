from os.path import dirname, abspath, join
import numpy as np
import sys
import warnings

test_dir = abspath(dirname(__file__))
data_dir = join(test_dir, 'data')
proj_dir = dirname(test_dir)
sys.path.append(test_dir) # fucking cursed
warnings.filterwarnings("ignore", category=DeprecationWarning) 

from ttig.mfid import symmetric_matrix_square_root


def test_symmetric_matrix_square_root():
    randgen = np.random.default_rng()
    x = randgen.standard_normal(size=(5, 5))
    x = (x + x.T) / 2 
    x = x + np.diag(np.asarray([1]*5)) * 4 # x needs to be symmetric _and_ positive-semi-definite
    x_squared = x @ x
    x_hat = symmetric_matrix_square_root(x_squared)
    assert np.allclose(x, x_hat), print(x, '\n=========\n', x_hat)
