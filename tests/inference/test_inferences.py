import sys
from inspect import getmembers, isclass
import numpy as np
from json import load

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
            assert hasattr(inf, '_serialize')

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

        def test_json(self, tmp_path):
            inf = cls()
            inf_file = tmp_path / f'{cls.__name__.lower()}.json'
            inf.save_json(inf_file)

            assert inf_file.is_file()
            assert inf_file.suffix == '.json'

            with open(inf_file, 'r') as fp:
                inf_info = load(fp)

            assert 'classname' in inf_info
            assert 'module' in inf_info
            assert 'kwargs' in inf_info

            assert inf_info['classname'] == cls.__name__
            assert inf_info['module'] == inf.__module__
            assert isinstance(inf_info['kwargs'], dict)

            inf_loaded = cls.load_json(inf_file)
            inf_loaded2 = Inference.load_json(inf_file)

            assert type(inf_loaded) is cls
            assert type(inf_loaded) is type(inf)
            assert type(inf_loaded) is type(inf_loaded2)

            assert inf_loaded is not inf
            assert inf_loaded is not inf_loaded2


    return (f'{Test.__name__}{cls.__name__}', Test)


for members_class in getmembers(sys.modules['harissa.inference'], isclass):
    name, group = _create_test_group(members_class[1])
    globals()[name] = group
