import pytest
import sys
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
