from pytest import fixture
import subprocess
import numpy as np
from harissa import NetworkModel, NetworkParameter
from harissa.inference import Hartree
from . import cmd_to_args

@fixture(scope='module')
def output_dir(tmp_path_factory):
    return tmp_path_factory.mktemp('outputs')

@fixture(scope='module')
def data_dir(tmp_path_factory):
    return tmp_path_factory.mktemp('data')

@fixture(scope='module')
def network():
    # Initialize model parameters with 4 genes
    p = NetworkParameter(4)

    # Set degradation rates
    p.degradation_rna[:] = 1.0
    p.degradation_protein[:] = 0.2

    p.burst_frequency_min[:] = 0.0 * p.degradation_rna
    p.burst_frequency_max[:] = 2.0 * p.degradation_rna

    # Set creation rates
    p.creation_rna[:] = p.degradation_rna * p.rna_scale() 
    p.creation_protein[:] = p.degradation_protein * p.protein_scale()

    # Set basal activities
    p.basal[1:] = -5.0

    # Set interactions
    p.interaction[0,1] = 10.0
    p.interaction[1,2] = 10.0
    p.interaction[1,3] = 10.0
    p.interaction[3,4] = 10.0
    p.interaction[4,1] = -10.0
    p.interaction[2,2] = 10.0
    p.interaction[3,3] = 10.0

    return p

@fixture(scope='module')
def dataset(network):
    # Initialize model
    model = NetworkModel(network)
    times = np.floor(np.linspace(0.0, 20.0, 3))
    C = 90
    n_cells_per_time_point = C // times.size # 100
    data = model.simulate_dataset(
        time_points = times, 
        n_cells=n_cells_per_time_point, 
        burn_in_duration=5.0
    )
    return data

@fixture(scope='module')
def network_file(network, data_dir):
    fname = data_dir / 'network.npz'
    network.save(fname)
    return fname

@fixture(scope='module')
def network_hartree_file(dataset, data_dir):
    model = NetworkModel(inference=Hartree())
    fname = data_dir / 'hartree.npz'
    model.fit(dataset).parameter.save(fname)
    return fname

@fixture(scope='module')
def dataset_file(dataset, data_dir):
    fname = data_dir / 'dataset.npz'
    dataset.save(fname)
    return fname

def test_help():
    process = subprocess.run(cmd_to_args('harissa dataset -h'))

    assert process.returncode == 0


def test_dataset(dataset_file, network_file, output_dir):
    output = output_dir / 'dataset.npz'
    params = f'{dataset_file} {network_file} -o {output}'
    process = subprocess.run(cmd_to_args(f'harissa dataset {params}'))

    assert process.returncode == 0
    assert output.is_file()

def test_dataset_hartree(dataset_file, network_hartree_file, output_dir):
    output = output_dir / 'dataset.npz'
    params = f'{dataset_file} {network_hartree_file} -o {output}'
    process = subprocess.run(
        cmd_to_args(f'harissa dataset {params}'), 
        capture_output=True,
        text=True
    )

    print(process.stderr)

    assert process.returncode == 0
    assert output.is_file()