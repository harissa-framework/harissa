import sys
from inspect import getmembers, isclass
import numpy as np

from harissa.core import Inference, NetworkParameter, Dataset

def _create_test_group(cls):
    class Test:
        def test_subclass(self):
            assert issubclass(cls, Inference)

        def test_instance(self):
            inf = cls()
            assert(hasattr(inf, 'run'))

        def test_run_output(self):
            inf = cls()
            data = Dataset(np.zeros(1), np.zeros((1, 2), dtype=np.uint))
            res = inf.run(data)

            n_genes_stim = data.count_matrix.shape[1]

            assert(isinstance(res, Inference.Result))
            
            assert(hasattr(res, 'parameter'))
            assert(isinstance(res.parameter, NetworkParameter))

            assert(res.parameter.n_genes_stim == n_genes_stim)

    return (f'{Test.__name__}_{cls.__name__}', Test)


for members_class in getmembers(sys.modules['harissa.inference'], isclass):
    test_name, test_group = _create_test_group(members_class[1])
    globals()[test_name] = test_group