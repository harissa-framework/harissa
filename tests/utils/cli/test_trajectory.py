from pytest import fixture
import subprocess
import numpy as np
from harissa.core import NetworkParameter, Simulation
from . import cmd_to_args

@fixture(scope='module')
def output_dir(tmp_path_factory):
    return tmp_path_factory.mktemp('outputs')

@fixture(scope='module')
def simulations_param_dir(tmp_path_factory):
    return tmp_path_factory.mktemp('simulations_param')

@fixture(scope='module')
def networks_dir(tmp_path_factory):
    return tmp_path_factory.mktemp('networks')

@fixture(scope='module')
def simulation_param_file(simulations_param_dir):
    time = np.linspace(0,10,100)
    fname = simulations_param_dir / 'default.npz'
    np.savez(fname, time_points=time)

    return fname

@fixture(scope='module')
def simulation_state_param_file(simulations_param_dir):
    time = np.linspace(0,10,100)
    initial_state = np.array([[0.0, 0.0, 0.0, 0.0], 
                              [1.0, 0.0, 0.05, 0.1]])
    fname = simulations_param_dir / 'initial_state.npz'
    np.savez(fname, time_points=time, initial_state=initial_state)
    return fname

@fixture(scope='module')
def network_file(networks_dir):
    p = NetworkParameter(3)
    # Basal gene activities
    p.basal[1] = 5
    p.basal[2] = 5
    p.basal[3] = 5
    # Inhibitions in cycle
    p.interaction[1,2] = -10
    p.interaction[2,3] = -10
    p.interaction[3,1] = -10

    # Degradation rates (per unit of time)
    p.degradation_rna[:] = 1.0
    p.degradation_protein[:] = 0.2

    p.burst_frequency_min[:] = 0.0 * p.degradation_rna
    p.burst_frequency_max[:] = 2.0 * p.degradation_rna

    # Creation rates
    p.creation_rna[:] = p.degradation_rna * p.rna_scale() 
    p.creation_protein[:] = p.degradation_protein * p.protein_scale()

    fname = networks_dir / 'repressilator.npz'
    p.save(fname)
    
    return fname

def test_help():
    process = subprocess.run(cmd_to_args('harissa trajectory -h'))

    assert process.returncode == 0

def test_trajectory_default(simulation_param_file, network_file, output_dir):
    output = output_dir / 'default.npz'
    params = f'{simulation_param_file} {network_file} -o {output}' 
    process = subprocess.run(
        cmd_to_args(
            f'harissa trajectory {params}'
        )
    )

    assert process.returncode == 0
    assert output.is_file()

def test_trajectory_default_txt(
        simulation_param_file, 
        network_file, 
        output_dir
    ):
    output = output_dir / 'default_txt'
    params = f'{simulation_param_file} {network_file} -o {output} -f txt' 
    process = subprocess.run(
        cmd_to_args(
            f'harissa trajectory {params}'
        )
    )

    assert process.returncode == 0
    assert output.is_dir()
    for fname in Simulation.Result.param_names:
        assert (output / f'{fname}.txt').is_file()


def test_trajectory_bursty(simulation_param_file, network_file, output_dir):
    output = output_dir / 'bursty.npz'
    params = f'{simulation_param_file} {network_file} -o {output}' 
    process = subprocess.run(
        cmd_to_args(
            f'harissa trajectory {params} bursty'
        )
    )

    assert process.returncode == 0
    assert output.is_file()

def test_trajectory_ode(simulation_state_param_file, network_file, output_dir):
    output = output_dir / 'ode.npz'
    params = f'{simulation_state_param_file} {network_file} -o {output}' 
    process = subprocess.run(
        cmd_to_args(
            f'harissa trajectory {params} ode'
        )
    )

    assert process.returncode == 0
    assert output.is_file()