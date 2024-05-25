import sys
from inspect import getmembers, isclass
import numpy as np

from harissa.core import Inference, NetworkParameter, Dataset
import harissa.inference

def _create_test_group(cls):
    class Test:
        def test_subclass(self):
            assert issubclass(cls, Inference)

        def test_instance(self):
            inf = cls()
            assert hasattr(inf, 'run')
            assert hasattr(inf, 'directed')

        def test_run_output(self):
            inf = cls()
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
            res = inf.run(data, NetworkParameter(count_matrix.shape[1] - 1))

            n_genes_stim = data.count_matrix.shape[1]

            assert isinstance(res, Inference.Result)
            
            assert hasattr(res, 'parameter')
            assert isinstance(res.parameter, NetworkParameter)

            assert res.parameter.n_genes_stim == n_genes_stim

    return (f'{Test.__name__}{cls.__name__}', Test)


for members_class in getmembers(sys.modules['harissa.inference'], isclass):
    name, group = _create_test_group(members_class[1])
    globals()[name] = group