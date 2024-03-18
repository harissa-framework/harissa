import numpy as np
from harissa.networks.cascade import cascade

def test_autoactiv():
    param = cascade(4, True)
    diagonal = np.diagonal(param.interaction)

    assert np.all(diagonal > 0.0)
    assert np.all(diagonal == diagonal[1])