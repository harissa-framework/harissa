from abc import ABC, abstractmethod
from harissa.core.parameter import NetworkParameter
from harissa.core.dataset import Dataset

class Inference(ABC):
    """
    Abstract class for inference methods.
    """
    class Result:
        """
        Inference result
        """
        def __init__(self, parameter: NetworkParameter, **kwargs) -> None:
            self.parameter: NetworkParameter = parameter
            for key, value in kwargs.items():
                setattr(self, key, value)

    @abstractmethod
    def run(self, data: Dataset) -> Result:
        raise NotImplementedError(
            f'{self.__class__.__name__} must only ' 
             'implement this function (run) and not use it.')
