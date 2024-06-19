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
            if not isinstance(parameter, NetworkParameter):
                raise TypeError(('parameter must be a NetworkParameter'
                                 f'(instead of {type(parameter)})'))
            self.parameter: NetworkParameter = parameter
            for key, value in kwargs.items():
                setattr(self, key, value)

        # Add load methods
        @classmethod
        def load(cls, path, load_extra=False):
            return Inference.Result(NetworkParameter.load(path))
        
        @classmethod
        def load_txt(cls, path, load_extra=False):
            return Inference.Result(NetworkParameter.load_txt(path))

        # Add a "save" methods
        def save_txt(self, path, save_extra=False):
            if save_extra:
                self.save_extra_txt(path)
            return self.parameter.save_txt(path)
        
        def save_extra_txt(self, path):
            return

        def save(self, path, save_extra=False):
            if save_extra:
                self.save_extra(path)
            return self.parameter.save(path)
        
        def save_extra(self, path):
            return

    @property
    @abstractmethod    
    def directed(self) -> bool:
        raise NotImplementedError(
            f'{self.__class__.__name__} must only ' 
             'implement this function (directed) and not use it.'
        )

    @abstractmethod
    def run(self, data: Dataset, param: NetworkParameter) -> Result:
        raise NotImplementedError(
            f'{self.__class__.__name__} must only ' 
             'implement this function (run) and not use it.'
        )
