import numpy as np
import pytest
from harissa.networks.random_tree import random_tree

def test_autoactiv():
    param = random_tree(4, autoactiv=True)
    diagonal = np.diagonal(param.interaction)

    assert np.all(diagonal > 0.0)
    assert np.all(diagonal == diagonal[1])

def test_weights():
    n_genes = 10
    w = np.ones((n_genes + 1, n_genes + 1))

    param = random_tree(n_genes, w)

    assert param.n_genes == n_genes

    
@pytest.mark.parametrize('n_genes,shape', [
    (3, (3,3)),
    (3, (4,3)),
    (3, (3,4)),
])
def test_weights_wrong_shape(n_genes, shape):
    w = np.ones(shape)

    with pytest.raises(ValueError):
        random_tree(n_genes, w)