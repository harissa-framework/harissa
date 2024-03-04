import pytest
import sys
from inspect import getmembers, isclass
import numpy as np

from harissa.core import Simulation, NetworkParameter

def _create_test_group(cls):
    class Test:
        def test_subclass(self):
            assert issubclass(cls, Simulation)

        def test_instance(self):
            sim = cls()
            assert(hasattr(sim, 'run'))

        def test_run_input_with_empty_network_parameter(self):
            sim = cls()
            with pytest.raises(AttributeError):
                sim.run(np.empty(1), np.empty((2, 1)), None)

        def test_run_output_type(self):
            sim = cls()
            n_genes = 2
            time_points = np.empty(1)
            res = sim.run(
                time_points,
                np.empty((2, n_genes + 1)), 
                NetworkParameter(n_genes)
            )
            assert(isinstance(res, Simulation.Result))
            for param_name in Simulation.Result.param_names: 
                assert(hasattr(res, param_name))

            assert res.time_points == time_points

    return (f'{Test.__name__}_{cls.__name__}', Test)

for members_class in getmembers(sys.modules['harissa.simulation'], isclass):
    test_name, test_group = _create_test_group(members_class[1])
    globals()[test_name] = test_group