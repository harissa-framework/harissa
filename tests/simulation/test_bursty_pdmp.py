import numpy as np
import pytest
from harissa.parameter import NetworkParameter
from harissa.simulation.bursty_pdmp.bursty_pdmp import BurstyPDMP, Simulation

def test_subclass():
    assert(issubclass(BurstyPDMP, Simulation))
    
def test_instance():
    sim = BurstyPDMP()
    assert(hasattr(sim, 'run'))

def test_run_input_with_empty_network_parameter():
    sim = BurstyPDMP()
    with pytest.raises(TypeError):
        sim.run(np.empty((2, 1)), np.empty(1), NetworkParameter())

def test_run_output_type():
    sim = BurstyPDMP()
    n_genes = 2
    res = sim.run(
        np.empty((2, n_genes + 1)), 
        np.empty(1), 
        NetworkParameter(n_genes)
    )
    assert(isinstance(res, Simulation.Result))
    assert(hasattr(res, 'time_points') 
           and hasattr(res, 'rna_levels') 
           and hasattr(res, 'protein_levels'))