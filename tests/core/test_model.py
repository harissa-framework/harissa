import numpy as np
import pytest
from inspect import getmembers

from harissa.core import (
    NetworkModel, 
    NetworkParameter, 
    Inference,
    Simulation, 
    Dataset
)
from harissa.inference import default_inference, Hartree
from harissa.simulation import default_simulation, BurstyPDMP, ApproxODE

@pytest.mark.parametrize('param,inf,sim', [
    (None, default_inference(), default_simulation()),
    (5, default_inference(), default_simulation()),
    (NetworkParameter(5), default_inference(), default_simulation()),
    (None, Hartree(), default_simulation()),
    (None, default_inference(), BurstyPDMP()),
    (None, default_inference(), ApproxODE()),
])
def test_init(param, inf, sim):
    model = NetworkModel(param, inference=inf, simulation=sim)

    if param is not None:
        assert isinstance(model.parameter, NetworkParameter)
        if isinstance(param, NetworkParameter):
            assert model.parameter is param
    else:
        assert model.parameter is None
            

    assert isinstance(model.inference, Inference)
    assert isinstance(model.simulation, Simulation) 

@pytest.mark.parametrize('param,inf,sim', [
    (5.0, default_inference(), default_simulation()),
    ('foo', default_inference(), default_simulation()),
    (None, 2, default_simulation()),
    (None, None, default_simulation()),
    (None, default_inference(), 2),
    (None, default_inference(), None),
])
def test_init_wrong_type(param, inf, sim):
    with pytest.raises(TypeError):
        NetworkModel(param, inference=inf, simulation=sim)

@pytest.mark.parametrize('param,inf,sim', [
    (NetworkParameter(2), default_inference(), default_simulation()),
    (NetworkParameter(3), Hartree(), default_simulation()),
    (NetworkParameter(4), default_inference(), BurstyPDMP()),
    (NetworkParameter(5), default_inference(), ApproxODE()),
])
def test_setters(param, inf, sim):
    model = NetworkModel()

    model.parameter = param
    model.inference = inf
    model.simulation = sim

    assert model.parameter is not None
    assert model.inference is not None
    assert model.simulation is not None

    assert isinstance(model.parameter, NetworkParameter)
    assert isinstance(model.inference, Inference)
    assert isinstance(model.simulation, Simulation)

    assert model.parameter is param
    assert model.inference is inf 
    assert model.simulation is sim 

@pytest.mark.parametrize('param,inf,sim', [
    (None, None, None),
    (5, 7.0, 'foo'),
    (default_simulation(), NetworkParameter(2), default_inference()),
    (default_inference(), default_simulation(), NetworkParameter(2))
], ids=[
    'None, None, None', 
    'int, float, str',
    'Simulation, NetworkParameter, Inference',
    'Inference, Simulation, NetworkParameter'
])
def test_setter_wrong_type(param, inf, sim):
    model = NetworkModel()
    with pytest.raises(TypeError):
        model.parameter = param
    
    with pytest.raises(TypeError):
        model.inference = inf

    with pytest.raises(TypeError):
        model.simulation = sim


def test_parameter_shortcuts():
    model = NetworkModel()

    assert model.parameter is None

    # parameter shortcuts
    props = [
        key
        for key, _ in getmembers(
            type(model), 
            lambda o: isinstance(o, property)
        ) if key not in ['parameter', 'inference', 'simulation']
    ]

    for p in props:
        with pytest.raises(AttributeError):
            getattr(model, p)


    model.parameter = NetworkParameter(4)

    for p in filter(lambda p: hasattr(model.parameter, p), props):
        assert np.array_equal(getattr(model, p), getattr(model.parameter, p))

    assert np.array_equal(model.inter, model.parameter.interaction)

@pytest.fixture
def dataset():
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
    return data

def test_fit(dataset):
    model = NetworkModel()
    res = model.fit(dataset)

    assert model.parameter is not None
    assert model.parameter is res.parameter

@pytest.mark.parametrize('data', [
    None,
    np.zeros((10, 5), dtype=np.int_),
    'foo',
    5
], ids=['None', 'ndarray', 'str', 'int'])
def test_fit_wrong_type(data):
    model = NetworkModel()
    with pytest.raises(TypeError):
        model.fit(data)

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

def test_simulate(network_parameter):
    model = NetworkModel()

    with pytest.raises(AttributeError):
        model.simulate(np.arange(10))
    
    model.parameter = network_parameter
    
    for time_points in [np.array(10.0), np.arange(10.0)]:
        res = model.simulate(time_points)

        assert np.all(res.stimulus_levels == 1.0)

    initial_state = np.zeros((2, model.n_genes_stim))
    state_copy = initial_state.copy()
    res = model.simulate(time_points, initial_state)

    assert np.all(res.stimulus_levels== 0.0)
    assert np.array_equal(initial_state, state_copy)

@pytest.mark.parametrize('time_points,initial_state', [
    (10.0, None),
    ([10.0], None),
    ((10.0), None),
    (np.array(10), 10),
    (np.array(10), np.zeros((3, 4))),
    (np.array(10), np.zeros((2, 3))),
    (np.array(10), np.zeros((3, 3, 3))),
], ids=[
    'float, None',
    'list, None',
    'tuple, None',
    'ndarray, int',
    'ndarray, ndarray (3, 4)',
    'ndarray, ndarray (2, 3)',
    'ndarray, ndarray (3, 3, 3)',
])
def test_simulate_wrong_type(network_parameter, time_points, initial_state):
    model = NetworkModel(network_parameter)

    with pytest.raises(TypeError):
        model.simulate(time_points, initial_state=initial_state)

@pytest.mark.parametrize('time_points,initial_time', [
    (np.array([[[10.0]]]), None),
    (np.array([[[[10.0]]]]), None),
    (np.array([10.0, 10.0]), None),
    (np.array([10.0, 5.0, 20.0]), None),
    (np.array(-1.0), None),
    (np.array(5.0), 10.0),
], ids=[
    '3D',
    '4D',
    'duplicate',
    'unsorted',
    'default initial_time > time_points',
    'initial_time > time_points'
])
def test_simulate_wrong_values(network_parameter, time_points, initial_time):
    model = NetworkModel(network_parameter)

    with pytest.raises(ValueError):
        if initial_time is None:
            model.simulate(time_points)
        else:
            model.simulate(time_points, initial_time=initial_time)

def test_burn_in(network_parameter):
    model = NetworkModel(network_parameter)

    state = model.burn_in(10.0)

    assert isinstance(state, np.ndarray)
    assert state.shape == (2, model.n_genes_stim)
    assert state[1, 0] == 1.0

def test_simulate_dataset(network_parameter):
    model = NetworkModel(network_parameter)

    time_points = np.arange(10)
    n_cells = 10

    # int
    dataset = model.simulate_dataset(time_points, n_cells)
    assert dataset.time_points.size == time_points.size * n_cells

    # list[int]
    n_cells = [10-i for i in range(10)]

    dataset = model.simulate_dataset(time_points, n_cells)
    assert dataset.time_points.size == 0.5 * len(n_cells) * (len(n_cells) + 1)

    # ndarray
    n_cells = np.array(n_cells, dtype=np.int_)

    dataset = model.simulate_dataset(time_points, n_cells)
    assert dataset.time_points.size == 0.5 * n_cells.size * (n_cells.size + 1)

@pytest.mark.parametrize('time_points,n_cells', [
    (5.0, 2),
    (np.array([5.0]), 2.0),
    (np.array([5.0, 0.0]), {0: 2}),
    ('foo', 2),
    (np.array([0.0, 5.0]), 'bar')
])
def test_simulate_dataset_wrong_type(network_parameter, time_points, n_cells):
    model = NetworkModel(network_parameter)

    with pytest.raises(TypeError):
        model.simulate_dataset(time_points, n_cells)


@pytest.mark.parametrize('time_points,n_cells', [
    (np.array([-5.0]), 2),
    (np.array([5.0, 0.0]), -2),
    (np.array([5.0, 0.0]), [1, 4, 3, 10]),
    (np.array([0.0, 5.0]), np.array([1, 0]))
])
def test_simulate_dataset_wrong_values(network_parameter,time_points,n_cells):
    model = NetworkModel(network_parameter)

    with pytest.raises(ValueError):
        model.simulate_dataset(time_points, n_cells)
    