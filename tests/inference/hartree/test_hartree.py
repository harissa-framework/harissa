import pytest
import sys
from pathlib import Path
import numpy as np
from harissa.core import NetworkParameter, Dataset, Inference
import harissa.inference.hartree.base as base
from importlib import reload

@pytest.fixture
def reload_base():
    if 'numba' in sys.modules:
        del sys.modules['numba']
    reload(base)

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
    return Dataset(time_points, count_matrix)

@pytest.fixture
def dataset_one():
    time_points = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
    count_matrix = np.array([
        # s g1 g2 g3
        [0, 1, 1, 1], # Cell 1
        [0, 1, 1, 1], # Cell 2
        [1, 1, 1, 1], # Cell 3
        [1, 1, 1, 1], # Cell 4
        [1, 1, 1, 1], # Cell 5
    ], dtype=np.uint)
    return Dataset(time_points, count_matrix)

class TestHartree:
    def test_use_numba_default(self, reload_base):
        inf = base.Hartree()
        assert inf.use_numba
        assert 'numba' in sys.modules
        assert base._infer_network_jit is not None
        assert inf._infer_network is base._infer_network_jit

    def test_use_numba_False(self, reload_base):
        inf = base.Hartree(use_numba=False)
        assert 'numba' not in sys.modules
        assert not inf.use_numba
        assert base._infer_network_jit is None
        assert inf._infer_network is base.infer_network

    def test_use_numba_True(self, reload_base):
        inf = base.Hartree(use_numba=True)
        assert inf.use_numba
        assert 'numba' in sys.modules
        assert base._infer_network_jit is not None
        assert inf._infer_network is base._infer_network_jit

    def test_use_numba_False_True_False(self, reload_base):
        inf = base.Hartree(use_numba=False)
        assert not inf.use_numba
        assert 'numba' not in sys.modules
        assert base._infer_network_jit is None
        assert inf._infer_network is base.infer_network


        inf.use_numba = True

        assert inf.use_numba
        assert 'numba' in sys.modules
        assert base._infer_network_jit is not None
        assert inf._infer_network is base._infer_network_jit

        inf.use_numba = False

        assert not inf.use_numba
        assert 'numba' in sys.modules
        assert base._infer_network_jit is not None
        assert inf._infer_network is base.infer_network

    def test_run_with_numba(self, dataset):
        from numba.core import config
        for disable_jit in [1, 0]:
            config.DISABLE_JIT = disable_jit
            reload(base)

            inf = base.Hartree(verbose= True, use_numba=True)
            res = inf.run(dataset)

            n_genes_stim = dataset.count_matrix.shape[1]

            assert isinstance(res, base.Hartree.Result)
            assert isinstance(res, Inference.Result)
            
            assert hasattr(res, 'parameter')
            assert isinstance(res.parameter, NetworkParameter)

            assert res.parameter.n_genes_stim == n_genes_stim

    def test_run_without_numba(self, dataset):
        inf = base.Hartree(verbose= True, use_numba=False)
        res = inf.run(dataset)

        n_genes_stim = dataset.count_matrix.shape[1]

        assert isinstance(res, base.Hartree.Result)
        assert isinstance(res, Inference.Result)
        
        assert hasattr(res, 'parameter')
        assert isinstance(res.parameter, NetworkParameter)

        assert res.parameter.n_genes_stim == n_genes_stim

    def test_dataset_one(self, dataset_one):
        inf = base.Hartree(tolerance= 1e-10, use_numba=False)
        with pytest.raises(RuntimeError):
            inf.run(dataset_one)

        assert True


def test_save_extra_txt(tmp_path, dataset):
    inf = base.Hartree(use_numba=False)
    res = inf.run(dataset)

    path = tmp_path / 'foo'

    res.save_txt(path, True)
    assert path.is_dir()

    path = path / 'extra'
    assert path.is_dir()

    for extra, ext in zip(
        ['basal_time', 'interaction_time', 'y'], 
        [None, None, '.txt']
        ):
        if ext is not None:
            assert (path / extra).with_suffix(ext).is_file()
        else:
            assert (path / extra).is_dir()


def test_save_extra(tmp_path, dataset):
    inf = base.Hartree(use_numba=False)
    res = inf.run(dataset)

    path = tmp_path / 'foo'

    res.save(path, True)

    assert path.with_suffix('.npz').is_file()
    for extra in ['basal_time', 'interaction_time', 'y']:
        assert Path(f'{path}_extra_{extra}').with_suffix('.npz').is_file()