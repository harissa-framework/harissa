from abc import ABC, abstractmethod
import numpy as np

from harissa.parameter import NetworkParameter

class Inference(ABC):
    class Result:
        """
        Result of inference
        """
        def __init__(self, parameter : NetworkParameter, **kwargs) -> None:
            self.parameter : NetworkParameter = parameter
            for key, value in kwargs.items():
                setattr(self, key, value)

    @abstractmethod
    def run(self, data: np.ndarray) -> Result:              
        raise NotImplementedError(
            f'{self.__class__.__name__} must only ' 
             'implement this function (run) and not use it.')