import pytest
import numpy as np
from harissa.utils.npz_io import (
    ParamInfos,
    load_dir,
    load_npz,
    save_dir,
    save_npz
)

@pytest.fixture
def param_names():
    return {
        'time_points': ParamInfos(True, np.float64, 1),
        'initial_state': ParamInfos(False, np.float64, 2),
        'count_matrix': ParamInfos(True, np.uint, 2),
        'gene_names': ParamInfos(False, np.str_, 1)
    }

@pytest.fixture(scope="module")
def arrays():
    return {
        'time_points' : np.random.rand(2),
        'count_matrix' : np.array([[1, 2], [3, 4]], dtype=np.uint),
        'gene_names': np.array(['foo', 'bar'], dtype= np.str_)
    }

@pytest.fixture(scope="module")
def npz_file(tmp_path_factory, arrays):
    path = tmp_path_factory.mktemp('data') / 'foo.npz'
    np.savez_compressed(path, **arrays)
    return path

@pytest.fixture(scope="module")
def txt_dir(tmp_path_factory, arrays):
    path = tmp_path_factory.mktemp('data') / 'foo'
    path.mkdir()
    for k, v in arrays.items():
        param = {
            'fname' : (path / k).with_suffix('.txt'),
            'X': v
        }
        if v.dtype.type is  np.str_:
            param['fmt'] = '%s'
        np.savetxt(**param)
    return path

class TestNPZ:
    def test_load(self, npz_file, param_names, arrays):
        data = load_npz(npz_file, param_names)

        for (k0, arr0), (k1, arr1) in zip(data.items(), arrays.items()):
            assert k0 == k1
            assert arr0.dtype == arr1.dtype
            assert np.array_equal(arr0, arr1)

    def test_load_not_found(self, tmp_path):
        with pytest.raises(RuntimeError):
            load_npz(tmp_path / 'foo', {})


    def test_load_unexpected_array(self, npz_file):
        with pytest.raises(RuntimeError):
            load_npz(npz_file, {'foo': ParamInfos(True, np.float64, 1)})

    def test_load_missing_array(self, tmp_path, arrays, param_names):
        npz_file = tmp_path / 'foo.npz'
        np.savez_compressed(npz_file, time_points=arrays['time_points'])
        with pytest.raises(RuntimeError):
            load_npz(npz_file, param_names)

    def test_save(self, tmp_path, arrays):
        path = save_npz(tmp_path / 'foo.npz', arrays)
        data = np.load(path)

        for (k0, arr0), (k1, arr1) in zip(data.items(), arrays.items()):
            assert k0 == k1
            assert arr0.dtype == arr1.dtype
            assert np.array_equal(arr0, arr1)



class TestDir:
    def test_load_not_found(self, tmp_path):
        with pytest.raises(RuntimeError):
            load_dir(tmp_path / 'foo', {})

    def test_load(self, txt_dir, param_names, arrays):
        data = load_dir(txt_dir, param_names)

        for (k0, arr0), (k1, arr1) in zip(data.items(), arrays.items()):
            assert k0 == k1
            assert arr0.dtype == arr1.dtype
            assert np.array_equal(arr0, arr1)

    def test_load_unexpected_array(self, txt_dir):
        with pytest.raises(RuntimeError):
            load_dir(txt_dir, {'foo': ParamInfos(True, np.float64, 1)})

    def test_load_missing_array(self, tmp_path, arrays, param_names):
        txt_dir = tmp_path / 'foo'
        txt_dir.mkdir()
        np.savetxt(txt_dir / 'time_points.txt', arrays['time_points'])
        with pytest.raises(RuntimeError):
            load_dir(txt_dir, param_names)

    def test_save(self, tmp_path, arrays):
        path = save_dir(tmp_path / 'foo', arrays)
        data = {
            'time_points' : np.loadtxt(path / 'time_points.txt'),
            'count_matrix' : np.loadtxt(path / 'count_matrix.txt', dtype=np.uint),
            'gene_names' : np.loadtxt(path / 'gene_names.txt', dtype=np.str_)
        }

        for (k0, arr0), (k1, arr1) in zip(data.items(), arrays.items()):
            assert k0 == k1
            assert arr0.dtype == arr1.dtype
            assert np.array_equal(arr0, arr1)
