import pytest
import sys
import numpy as np
from harissa.core import NetworkParameter, Simulation
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

def test_use_numba_True_False_True(reload_base):
    sim = base.ApproxODE(use_numba=True)
    assert sim.use_numba
    assert 'numba' in sys.modules
    assert base._simulation_jit is not None
    assert sim._simulation is base._simulation_jit

    sim.use_numba = False

    assert not sim.use_numba
    assert 'numba' in sys.modules
    assert base._simulation_jit is not None
    assert sim._simulation is base.simulation

    sim.use_numba = True

    assert sim.use_numba
    assert 'numba' in sys.modules
    assert base._simulation_jit is not None
    assert sim._simulation is base._simulation_jit

def test_run_with_numba():
    sim = base.ApproxODE(use_numba=True)

    param = NetworkParameter(3)
    param.degradation_rna[:] = 1
    param.degradation_protein[:] = 0.2
    param.basal[1] = 5
    param.basal[2] = 5
    param.basal[3] = 5
    param.interaction[1,2] = -10
    param.interaction[2,3] = -10
    param.interaction[3,1] = -10
    scale = param.burst_size_inv / param.burst_frequency_max
    param.creation_rna[:] = param.degradation_rna * scale 
    param.creation_protein[:] = param.degradation_protein * scale
    
    time_points = np.arange(10)
    initial_state = np.zeros((2, param.n_genes_stim))
    initial_state[1, 0] = 1

    res = sim.run(time_points, initial_state, param)

    assert isinstance(res, Simulation.Result)