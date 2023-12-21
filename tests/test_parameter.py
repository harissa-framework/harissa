import pytest
import numpy as np
from harissa import NetworkParameter

def test_init_neg():
    with pytest.raises(ValueError):
        NetworkParameter(0)

    with pytest.raises(ValueError):
        NetworkParameter(-1)

def test_init_not_int():
    with pytest.raises(TypeError):
        NetworkParameter(2.5)

def test_init():
    n_genes = 1
    param = NetworkParameter(n_genes)

    assert param.n_genes == n_genes
    assert param.n_genes_stim == n_genes + 1
    
    for array_name in param._array_names():
        array = getattr(param, array_name)

        if array.ndim == 2:
            assert array.shape[1] == param.n_genes_stim
            assert np.all(array.mask[:, 0])
        else:
            assert array.size == param.n_genes_stim
            assert array.mask[0]

    assert param.interaction.shape[0] == param.n_genes_stim
    

def test_setters():
    param = NetworkParameter(1)

    with pytest.raises(TypeError):
        param.basal = 2
    
    with pytest.raises(ValueError):
        param.basal = np.empty((param.n_genes))

    with pytest.raises(ValueError):
        param.basal = np.empty((param.n_genes_stim + 1))

    with pytest.raises(ValueError):
        param.basal = np.empty((param.n_genes_stim), dtype='bool')

    param.basal = np.ones((param.n_genes_stim))

    assert param.basal.data[0] == 0.0
    assert param.basal[1] == 1.0