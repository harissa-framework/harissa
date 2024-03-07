import pytest
import sys
import numpy as np
from harissa.core import NetworkParameter, Dataset, Inference
import harissa.inference.hartree.base as base
from importlib import reload

@pytest.fixture
def reload_base():
    if 'numba' in sys.modules:
        del sys.modules['numba']
    reload(base)

def test_use_numba_default(reload_base):
    inf = base.Hartree()
    assert inf.use_numba
    assert 'numba' in sys.modules
    assert base._infer_network_jit is not None
    assert inf._infer_network is base._infer_network_jit

def test_use_numba_False(reload_base):
    inf = base.Hartree(use_numba=False)
    assert 'numba' not in sys.modules
    assert not inf.use_numba
    assert base._infer_network_jit is None
    assert inf._infer_network is base.infer_network

def test_use_numba_True(reload_base):
    inf = base.Hartree(use_numba=True)
    assert inf.use_numba
    assert 'numba' in sys.modules
    assert base._infer_network_jit is not None
    assert inf._infer_network is base._infer_network_jit

def test_use_numba_False_True_False(reload_base):
    inf = base.Hartree(use_numba=False)
    assert not inf.use_numba
    assert 'numba' not in sys.modules
    assert base._infer_network_jit is None
    assert inf._infer_network is base.infer_network


    inf.use_numba = True

    assert inf.use_numba
    assert 'numba' in sys.modules
    assert base._infer_network_jit is not None
    assert inf._infer_network is base._infer_network_jit

    inf.use_numba = False

    assert not inf.use_numba
    assert 'numba' in sys.modules
    assert base._infer_network_jit is not None
    assert inf._infer_network is base.infer_network


def test_run_without_numba():
    inf = base.Hartree(use_numba=False)
    time_points = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
    count_matrix = np.array([
        # s g1 g2 g3
        [0, 4, 1, 0], # Cell 1
        [0, 5, 0, 1], # Cell 2
        [1, 1, 2, 4], # Cell 3
        [1, 2, 0, 8], # Cell 4
        [1, 0, 0, 3], # Cell 5
    ], dtype=np.uint)
    data = Dataset(time_points, count_matrix)
    res = inf.run(data)

    n_genes_stim = data.count_matrix.shape[1]

    assert isinstance(res, base.Hartree.Result)
    assert isinstance(res, Inference.Result)
    
    assert hasattr(res, 'parameter')
    assert isinstance(res.parameter, NetworkParameter)

    assert res.parameter.n_genes_stim == n_genes_stim