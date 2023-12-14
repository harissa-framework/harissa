import pytest
import numpy as np
from harissa.inference.inference import Inference

def test_inference_instance():
    with pytest.raises(TypeError):
        Inference()

class InferenceMissingRun(Inference):
    def __init__(self):
        ...

def test_inference_missing_run():
    with pytest.raises(TypeError):
        InferenceMissingRun()

class InferenceSuperRun(Inference):
    def __init__(self):
        ...

    def run(self, data: np.ndarray) -> Inference.Result:
        return super().run(data)
    
def test_inference_super_run():
    inf = InferenceSuperRun()

    with pytest.raises(NotImplementedError):
        inf.run(np.empty(1))