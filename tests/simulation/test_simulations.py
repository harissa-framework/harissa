import pytest
import sys
from inspect import getmembers, isclass
import numpy as np

from harissa.core import Simulation, NetworkParameter
import harissa.simulation

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
                sim.run(np.empty(1), np.empty((2, 1)), None)

        def test_run_output_type(self):
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
            
            time_points = np.arange(10, dtype=np.float_)
            initial_state = np.zeros((2, param.n_genes_stim))
            initial_state[1, 0] = 1
            
            for tp in [time_points, np.array([time_points[-1]])]:
                res = cls().run(tp, initial_state, param)

                assert isinstance(res, Simulation.Result)
                assert res.rna_levels.shape[1] == param.n_genes_stim

    return (f'{Test.__name__}{cls.__name__}', Test)

for members_class in getmembers(sys.modules['harissa.simulation'], isclass):
    name, group = _create_test_group(members_class[1])
    globals()[name] = group