import pytest
import sys
import harissa.simulation.approx_ode.base as base
from importlib import reload

@pytest.fixture
def reload_base():
    if 'numba' in sys.modules:
        del sys.modules['numba']
    reload(base)

def test_use_numba_default(reload_base):
    sim = base.ApproxODE()
    assert not sim.use_numba
    assert 'numba' not in sys.modules
    assert base._simulation_jit is None
    assert sim._simulation is base.simulation

def test_use_numba_False(reload_base):
    sim = base.ApproxODE(use_numba=False)
    assert 'numba' not in sys.modules
    assert not sim.use_numba
    assert base._simulation_jit is None
    assert sim._simulation is base.simulation


def test_use_numba_True(reload_base):
    sim = base.ApproxODE(use_numba=True)
    assert sim.use_numba
    assert 'numba' in sys.modules
    assert base._simulation_jit is not None
    assert sim._simulation is base._simulation_jit


def test_use_numba_False_True_False(reload_base):
    sim = base.ApproxODE(use_numba=False)
    assert not sim.use_numba
    assert 'numba' not in sys.modules
    assert base._simulation_jit is None
    assert sim._simulation is base.simulation

    sim.use_numba = True

    assert sim.use_numba
    assert 'numba' in sys.modules
    assert base._simulation_jit is not None
    assert sim._simulation is base._simulation_jit

    sim.use_numba = False

    assert not sim.use_numba
    assert 'numba' in sys.modules
    assert base._simulation_jit is not None
    assert sim._simulation is base.simulation
