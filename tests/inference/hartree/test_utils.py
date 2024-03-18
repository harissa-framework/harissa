import numpy as np
import pytest
import harissa.inference.hartree.utils as utils

@pytest.mark.parametrize('x', [
    np.array([4, 5, 1, 2, 0]),
    np.zeros(5),
    np.ones(5)
])
def test_estimate_gamma(x):
    y = utils.estimate_gamma(x)

    assert y[0] >= 0
    assert y[1] > 0

@pytest.mark.parametrize('x', [
    np.array([4, 5, 1, 2, 0]),
    np.zeros(5),
    np.ones(5),
    np.array([1, 2])
])
def test_estimate_gamma_poisson(x):
    y = utils.estimate_gamma_poisson(x)

    assert y[0] >= 0
    assert y[1] > 0

@pytest.mark.parametrize('x', [
    np.array([4, 5, 1, 2, 0]),
    np.zeros(5),
    np.ones(5),
    np.array([1, 2])
])
def test_transform(x):
    y = utils.transform(x)

    assert np.all(y >= 0)

