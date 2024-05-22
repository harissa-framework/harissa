from pytest import fixture
import subprocess
import numpy as np
from harissa import NetworkModel, NetworkParameter

from . import cmd_to_args

@fixture(scope='module')
def output_dir(tmp_path_factory):
    return tmp_path_factory.mktemp('outputs')

@fixture(scope='module')
def datasets_dir(tmp_path_factory):
    return tmp_path_factory.mktemp('datasets')

@fixture(scope='module')
def networks_dir(tmp_path_factory):
    return tmp_path_factory.mktemp('networks')

@fixture(scope='module')
def network_param():
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
def network_file(networks_dir, network_param):
    return network_param.save(networks_dir / 'network.npz')


@fixture(scope='module')
def dataset_file(datasets_dir, network_param):
    # Initialize model
    model = NetworkModel(network_param)
    times = np.floor(np.linspace(0.0, 20.0, 3))
    C = 90
    n_cells_per_time_point = C // times.size 
    data = model.simulate_dataset(
        time_points = times, 
        n_cells=n_cells_per_time_point, 
        burn_in_duration=5.0
    )

    # Save data in basic format
    return data.save(datasets_dir / 'dataset.npz')

def test_help():
    process = subprocess.run(cmd_to_args('harissa infer -h'))

    assert process.returncode == 0

def test_infer_default(dataset_file, network_file, output_dir):
    output = output_dir / 'default.npz'
    cmd = f'harissa infer {dataset_file} -n {network_file} -o {output}' 
    process = subprocess.run(cmd_to_args(cmd))

    assert process.returncode == 0
    assert output.is_file()

def test_infer_hartree(dataset_file, network_file, output_dir):
    output = output_dir / 'hartree.npz'
    cmd = f'harissa infer {dataset_file} -n {network_file} -o {output} hartree'
    process = subprocess.run(cmd_to_args(cmd))

    assert process.returncode == 0
    assert output.is_file()

def test_infer_cardamom(dataset_file, network_file, output_dir):
    output = output_dir / 'cardamom.npz'
    cmd= f'harissa infer {dataset_file} -n {network_file} -o {output} cardamom'
    process = subprocess.run(cmd_to_args(cmd))

    assert process.returncode == 0
    assert output.is_file()