import numpy as np
import pytest
import harissa.inference.cardamom.utils as utils

@pytest.mark.parametrize('x', [
    np.array([4, 5, 1, 2, 0]),
    np.zeros(5),
    np.ones(5),
    np.array([1, 2])
])
def test_estim_gamma_poisson(x):
    y = utils.estim_gamma_poisson(x)

    assert y[0] >= 0
    assert y[1] > 0