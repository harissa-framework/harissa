import numpy as np
from harissa import NetworkParameter, Inference
from harissa.inference import Hartree

def test_subclass():
    assert(issubclass(Hartree, Inference))
    
def test_instance():
    inf = Hartree()
    assert(hasattr(inf, 'run'))

def test_run_output_type():
    inf = Hartree()
    res = inf.run(np.empty((1, 2)))

    assert(isinstance(res, Inference.Result))
    assert(hasattr(res, 'parameter'))
    assert(isinstance(res.parameter, NetworkParameter))