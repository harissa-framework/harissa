# import numpy as np
# import pytest
# from harissa.core import NetworkParameter, Simulation
# from harissa.simulation import ApproxODE

# def test_subclass():
#     assert(issubclass(ApproxODE, Simulation))
    
# def test_instance():
#     sim = ApproxODE()
#     assert(hasattr(sim, 'run'))

# def test_run_input_with_empty_network_parameter():
#     sim = ApproxODE()
#     with pytest.raises(AttributeError):
#         sim.run(np.empty(1), np.empty((2, 1)), None)

# def test_run_output_type():
#     sim = ApproxODE()
#     n_genes = 2
#     res = sim.run(
#         np.empty(1),
#         np.empty((2, n_genes + 1)), 
#         NetworkParameter(n_genes)
#     )
#     assert(isinstance(res, Simulation.Result))
#     assert(hasattr(res, 'time_points') 
#            and hasattr(res, 'rna_levels') 
#            and hasattr(res, 'protein_levels'))