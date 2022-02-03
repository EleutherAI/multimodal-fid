
import torch
from ttig.models.model import spherical_dist_loss


def test_spherical_dist_loss():
    x = torch.as_tensor([[1., 0., 0., 1., -1.]])
    y = torch.as_tensor([[-1., -1., 0., 0., 1.]])
    loss = spherical_dist_loss(x, y)
    assert loss.shape == (1,)
    x = torch.randn((10, 5))
    y = torch.randn((10, 5))
    assert spherical_dist_loss(x, y).shape == (10,)

