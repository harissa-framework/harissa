import numpy as np
from harissa.parameter import NetworkParameter
from harissa.inference.hartree.hartree import Hartree, Inference

def test_subclass():
    assert(issubclass(Hartree, Inference))
    
def test_instance():
    inf = Hartree()
    assert(hasattr(inf, 'run'))

def test_run_output_type():
    inf = Hartree()
    res = inf.run(np.empty((1, 1)))

    assert(isinstance(res, Inference.Result))
    assert(hasattr(res, 'parameter'))
    assert(isinstance(res.parameter, NetworkParameter))