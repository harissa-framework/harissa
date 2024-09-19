import pytest
from inspect import getmembers, isclass
import numpy as np
from harissa.core import Simulation, NetworkParameter
import harissa.simulation

@pytest.fixture
def network_param():
    p = NetworkParameter(3)
    p.degradation_rna[:] = 1
    p.degradation_protein[:] = 0.2
    p.basal[1] = 5
    p.basal[2] = 5
    p.basal[3] = 5
    p.interaction[1,2] = -10
    p.interaction[2,3] = -10
    p.interaction[3,1] = -10
    p.creation_rna[:] = p.degradation_rna * p.rna_scale()
    p.creation_protein[:] = p.degradation_protein * p.protein_scale()

    return p

def _create_test_group(cls):
    class Test:
        def test_subclass(self):
            assert issubclass(cls, Simulation)

        def test_instance(self):
            sim = cls()
            assert hasattr(sim, 'run')

        def test_run_input_with_empty_network_parameter(self):
            sim = cls()
            with pytest.raises(AttributeError):
                sim.run(np.empty(1), np.empty((2, 1)), np.empty(1), None)

        def test_run_output_type(self, network_param):
            time_points = np.arange(10, dtype=np.float64)
            initial_state = np.zeros((2, network_param.n_genes_stim))
            initial_state[1, 0] = 1

            for tp in [time_points, np.array([time_points[-1]])]:
                stimulus = np.ones(tp.shape)
                res = cls().run(tp, initial_state, stimulus, network_param)

                assert isinstance(res, Simulation.Result)
                assert res.rna_levels.shape[0] == tp.size
                assert res.rna_levels.shape[1] == network_param.n_genes_stim

        def test_stimulus(self, network_param):
            time_points = np.array([5.0, 10.0, 15.0])
            initial_state = np.zeros((2, network_param.n_genes_stim))
            stimulus = np.array([1.0, 0.0, 1.0])

            res = cls().run(
                time_points,
                initial_state,
                stimulus,
                network_param
            )

            assert isinstance(res, Simulation.Result)
            assert res.rna_levels.shape[0] == time_points.size
            assert res.rna_levels.shape[1] == network_param.n_genes_stim
            assert np.array_equal(res.stimulus_levels, stimulus)

    return (f'{Test.__name__}{cls.__name__}', Test)

for cls in set(map(
    lambda member_class : member_class[1],
    getmembers(harissa.simulation, isclass)
)):
    name, group = _create_test_group(cls)
    globals()[name] = group
