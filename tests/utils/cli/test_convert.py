from pytest import fixture
import subprocess
import numpy as np
from harissa import Dataset
from . import cmd_to_args

@fixture(scope='module')
def outputs_dir(tmp_path_factory):
    return tmp_path_factory.mktemp('outputs')

@fixture(scope='module')
def data_dir(tmp_path_factory):
    return tmp_path_factory.mktemp('data')

@fixture(scope='module')
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

@fixture(scope='module')
def dataset_npz(dataset, data_dir):
    fname = data_dir / 'dataset.npz'
    dataset.save(fname)
    return fname

@fixture(scope='module')
def dataset_dir(dataset, data_dir):
    fname = data_dir / 'dataset'
    dataset.save_txt(fname)
    return fname

@fixture(scope='module')
def dataset_old(dataset, data_dir):
    data = dataset.count_matrix.astype(np.int_)
    data[:, 0] = dataset.time_points.astype(np.int_)
    fname = data_dir / 'dataset.txt'
    np.savetxt(fname, data)
    return fname


def test_help():
    process = subprocess.run(cmd_to_args('harissa convert -h'))

    assert process.returncode == 0


def test_convert_npz_to_txt(dataset_npz, outputs_dir):
    output = outputs_dir / 'dataset'
    process = subprocess.run(
        cmd_to_args(f'harissa convert {dataset_npz} {output}')
    )

    assert process.returncode == 0
    assert output.is_dir()
    for fname in Dataset.load(dataset_npz).as_dict():
        assert (output / f'{fname}.txt').is_file()

def test_convert_txt_to_npz(dataset_dir, outputs_dir):
    output = outputs_dir / 'dataset.npz'
    process = subprocess.run(
        cmd_to_args(f'harissa convert {dataset_dir} {output}')
    )

    assert process.returncode == 0
    assert output.is_file()

def test_convert_old_txt_to_npz(dataset_old, outputs_dir):
    output = outputs_dir / 'dataset_old.npz'
    process = subprocess.run(
        cmd_to_args(f'harissa convert {dataset_old} {output}')
    )

    assert process.returncode == 0
    assert output.is_file()

def test_convert_old_txt_to_txt(dataset_old, outputs_dir):
    output = outputs_dir / 'dataset_old'
    process = subprocess.run(
        cmd_to_args(f'harissa convert {dataset_old} {output}')
    )

    assert process.returncode == 0
    assert output.is_dir()
    for fname in Dataset.load_txt(dataset_old).as_dict():
        assert (output / f'{fname}.txt').is_file()