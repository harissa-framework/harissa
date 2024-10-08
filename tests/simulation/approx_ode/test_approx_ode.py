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

@pytest.fixture
def time_points():
    return np.arange(10, dtype=np.float64)

@pytest.fixture
def network_parameter():
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

    return param

@pytest.fixture
def initial_state(network_parameter):
    state = np.zeros((2, network_parameter.n_genes_stim))
    state[1, 0] = 1

    return state

def test_use_numba_default(reload_base):
    sim = base.ApproxODE()
    assert not sim.use_numba
    assert 'numba' not in sys.modules
    assert base._numba_functions[True] is None
    for name, f in base._numba_functions[False].items():
        assert vars(base)[name] is f

def test_use_numba_False(reload_base):
    sim = base.ApproxODE(use_numba=False)
    assert 'numba' not in sys.modules
    assert not sim.use_numba
    assert base._numba_functions[True] is None
    for name, f in base._numba_functions[False].items():
        assert vars(base)[name] is f


def test_use_numba_True(reload_base):
    sim = base.ApproxODE(use_numba=True)
    assert sim.use_numba
    assert 'numba' in sys.modules
    assert base._numba_functions[True] is not None
    for name, f in base._numba_functions[False].items():
        assert vars(base)[name] is f


def test_use_numba_False_True_False(reload_base):
    sim = base.ApproxODE(use_numba=False)
    assert not sim.use_numba
    assert 'numba' not in sys.modules
    assert base._numba_functions[True] is None
    for name, f in base._numba_functions[False].items():
        assert vars(base)[name] is f

    sim.use_numba = True

    assert sim.use_numba
    assert 'numba' in sys.modules
    assert base._numba_functions[True] is not None
    for name, f in base._numba_functions[False].items():
        assert vars(base)[name] is f

    sim.use_numba = False

    assert not sim.use_numba
    assert 'numba' in sys.modules
    assert base._numba_functions[True] is not None
    for name, f in base._numba_functions[False].items():
        assert vars(base)[name] is f

def test_use_numba_True_False_True(reload_base):
    sim = base.ApproxODE(use_numba=True)
    assert sim.use_numba
    assert 'numba' in sys.modules
    assert base._numba_functions[True] is not None
    for name, f in base._numba_functions[False].items():
        assert vars(base)[name] is f

    sim.use_numba = False

    assert not sim.use_numba
    assert 'numba' in sys.modules
    assert base._numba_functions[True] is not None
    for name, f in base._numba_functions[False].items():
        assert vars(base)[name] is f

    sim.use_numba = True

    assert sim.use_numba
    assert 'numba' in sys.modules
    assert base._numba_functions[True] is not None
    for name, f in base._numba_functions[False].items():
        assert vars(base)[name] is f

def test_run_with_numba(time_points, initial_state, network_parameter):
    from numba.core import config
    for disable_jit in [1, 0]:
        config.DISABLE_JIT = disable_jit
        reload(base)

        sim = base.ApproxODE(verbose=True, use_numba=True)
        for tp in [time_points, np.array([time_points[-1]])]:
            stimulus = np.ones(tp.shape)
            res = sim.run(tp, initial_state, stimulus, network_parameter)

            assert isinstance(res, Simulation.Result)


def test_run_without_numba(time_points, initial_state, network_parameter):
    sim = base.ApproxODE(verbose=True, use_numba=False)
    stimulus = np.ones(time_points.shape)
    res = sim.run(time_points, initial_state, stimulus, network_parameter)

    assert isinstance(res, Simulation.Result)
