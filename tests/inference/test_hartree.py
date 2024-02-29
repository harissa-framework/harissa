# import numpy as np
# from harissa.core import NetworkParameter, Inference, Dataset
# from harissa.inference import Hartree

# def test_subclass():
#     assert(issubclass(Hartree, Inference))
    
# def test_instance():
#     inf = Hartree()
#     assert(hasattr(inf, 'run'))

# def test_run_output_type():
#     inf = Hartree()
#     res = inf.run(Dataset(np.zeros(1), np.zeros((1, 2), dtype=np.uint)))

#     assert(isinstance(res, Inference.Result))
#     assert(hasattr(res, 'parameter'))
#     assert(isinstance(res.parameter, NetworkParameter))