from pytest import fixture
import subprocess
import numpy as np
from harissa import NetworkModel, NetworkParameter
from . import cmd_to_args

@fixture(scope='module')
def outputs_dir(tmp_path_factory):
    return tmp_path_factory.mktemp('outputs')

@fixture(scope='module')
def datasets_dir(tmp_path_factory):
    return tmp_path_factory.mktemp('datasets')

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
def times():
    return np.floor(np.linspace(0.0, 20.0, 3))

@fixture(scope='module')
def C():
    return 90

@fixture(scope='module')
def ref_dataset(network, times, C):
    # Initialize model
    model = NetworkModel(network)
    n_cells_per_time_point = C // times.size
    data = model.simulate_dataset(
        time_points = times, 
        n_cells=n_cells_per_time_point, 
        burn_in_duration=5.0
    )
    return data

@fixture(scope='module')
def ref_dataset_file(ref_dataset, datasets_dir):
    fname = datasets_dir / 'ref_dataset.npz'
    ref_dataset.save(fname)
    return fname

@fixture(scope='module')
def dataset_file(network, datasets_dir, times, C):
    # Initialize model
    model = NetworkModel(network)
    n_cells_per_time_point = C // times.size # 100
    data = model.simulate_dataset(
        time_points = times, 
        n_cells=n_cells_per_time_point, 
        burn_in_duration=5.0
    )
    fname = datasets_dir / 'dataset.npz'
    data.save(fname)
    return fname

def test_help():
    process = subprocess.run(cmd_to_args('harissa visualize -h'))

    assert process.returncode == 0

def test_visualize_id(ref_dataset_file, outputs_dir):
    output = outputs_dir / 'id'
    params = f'{ref_dataset_file} {ref_dataset_file} -o {output} -dpu'
    process = subprocess.run(
        cmd_to_args(f'harissa visualize {params}')
    )

    pdfs = ['comparison.pdf', 'marginals.pdf']
    try:
        import umap
        pdfs = pdfs + ['umap.pdf']
    except ImportError:
        pass

    assert process.returncode == 0
    assert output.is_dir()
    for fname in pdfs:
        (output / fname).is_file()

def test_visualize(ref_dataset_file, dataset_file, outputs_dir):
    output = outputs_dir / 'id'
    params = f'{ref_dataset_file} {dataset_file} -o {output} -dpu'
    process = subprocess.run(
        cmd_to_args(f'harissa visualize {params}')
    )

    pdfs = ['comparison.pdf', 'marginals.pdf']
    try:
        import umap
        pdfs = pdfs + ['umap.pdf']
    except ImportError:
        pass

    assert process.returncode == 0
    assert output.is_dir()
    for fname in pdfs:
        (output / fname).is_file()