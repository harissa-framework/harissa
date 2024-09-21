"""
Abstract class for inferences
"""
from abc import ABC, abstractmethod
from json import dump, load
from pathlib import Path
from importlib import import_module
from typing import Union, Dict
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

        @classmethod
        def load_json(cls, path, load_extra=False):
            return Inference.Result(NetworkParameter.load_json(path))

        # Add a "save" methods
        def save(self, path, save_extra=False):
            path = self.parameter.save(path)
            if save_extra:
                self.save_extra(path)
            return path

        def save_extra(self, path):
            return

        def save_txt(self, path, save_extra=False):
            path = self.parameter.save_txt(path)
            if save_extra:
                self.save_extra_txt(path)
            return path

        def save_extra_txt(self, path):
            return

        def save_json(self, path, save_extra=False):
            path = self.parameter.save_json(path)
            if save_extra:
                self.save_extra_json(path)
            return path

        def save_extra_json(self, path):
            return

    def __init__(self, *init_keys) -> None:
        self._init_keys = init_keys

    @classmethod
    def load_json(cls, path: Union[str, Path]):
        path = Path(path).with_suffix('.json')
        with open(path, 'r') as fp:
            inference_info = load(fp)

        mod = import_module(inference_info['module'])
        inference_class = vars(mod)[inference_info['classname']]

        if not issubclass(inference_class, Inference):
            raise RuntimeError(
                f'{inference_class} must be a subclass of {cls}.'
            )

        return inference_class(
            **inference_class._deserialize(inference_info['kwargs'])
        )

    @classmethod
    def _deserialize(cls, kwargs: Dict) -> Dict:
        return kwargs

    def save_json(self, path: Union[str, Path]) -> Path:
        inference_info = {
            'classname' : self.__class__.__name__,
            'module': self.__module__,
            'kwargs': self._serialize()
        }
        path = Path(path).with_suffix('.json')
        with open(path, 'w') as fp:
            dump(inference_info, fp, indent=4)

        return path

    def _serialize(self) -> Dict:
        return {k:getattr(self, k) for k in self._init_keys}


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
