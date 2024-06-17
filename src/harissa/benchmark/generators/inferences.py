from typing import (
    Dict, 
    List,
    Tuple, 
    Callable, 
    Union, 
    Optional,
    TypeAlias
)
from collections.abc import Iterator

from dataclasses import dataclass
from pathlib import Path
from dill import loads, dumps

import matplotlib.colors
import matplotlib.pyplot
import numpy as np
import numpy.typing as npt
import matplotlib

from harissa.utils.progress_bar import alive_bar
from harissa.benchmark.generators.generic import GenericGenerator
from harissa.core import Inference
from harissa.inference import Hartree, Cardamom, Pearson

@dataclass
class InferenceInfo:
    inference: Union[Inference, Callable[[], Inference]]
    colors: npt.NDArray

K: TypeAlias = str
V: TypeAlias = Inference
class InferencesGenerator(GenericGenerator[K, V]):
    _inferences : Dict[str, InferenceInfo] = {}
    color_map: matplotlib.colors.Colormap = matplotlib.pyplot.get_cmap('tab20')

    def __init__(self,
        include: Optional[List[str]] = None, 
        exclude: Optional[List[str]] = None,
        path: Optional[Union[str, Path]] = None,
        verbose: bool = False
    ) -> None:
        super().__init__('inferences', include, exclude, path, verbose)

    @classmethod
    def register(cls, name: str, inference_info: InferenceInfo) -> None:
        if name not in cls._inferences:
            if isinstance(inference_info.inference, (Inference, Callable)):
                cls._inferences[name] = inference_info
            else:
                raise TypeError(('inference_callable must be a callable '
                             'that returns a Inference sub class.'))
        else:
            raise ValueError((f'{name} is already taken. '
                              f'Cannot register {inference_info}.'))
        
    @classmethod
    def register_defaults(cls) -> None:
        cls.register('Hartree', InferenceInfo(
            Hartree,
            np.array([cls.color_map(6), cls.color_map(7)])
        ))
        cls.register('Cardamom', InferenceInfo(
            Cardamom, 
            np.array([cls.color_map(8), cls.color_map(9)])
        ))
        cls.register('Pearson', InferenceInfo(
            Pearson,
            np.array([cls.color_map(14), cls.color_map(15)])
        ))

    @classmethod
    def unregister_all(cls) -> None:
        cls._networks = {}

    # Alias
    @property
    def inferences(self):
        return self.items

    @classmethod
    def available_inferences(cls) -> List[str]:
        return list(cls._inferences.keys())
    
    @classmethod
    def getInferenceInfo(cls, name: str) -> InferenceInfo:
        return cls._inferences[name]
    
    def _load_value(self, path: Path, key: K) -> V:
        inference_path = path / self.sub_directory_name / f'{key}.npz'
        with np.load(inference_path) as data:
            inf = loads(data['inference'].item())
            if not isinstance(inf, Inference):
                raise RuntimeError(
                    f'{inf} is not an Inference object.'
                )
        
        return inf

    def _generate_value(self, key: K) -> V:
        inf_info = self._inferences[key]
        if isinstance(inf_info.inference, Inference):
            inf = inf_info.inference
        else:
            inf = inf_info.inference()
        
        if not isinstance(inf, Inference):
            raise RuntimeError(
                (f'{inf_info.inference} is not an Inference subclass.')
            )
        
        # create a wrapper around inference method that is also an Inference
        BenchmarkInf = type(
            f'{type(inf).__name__}',
            (type(inf),),
            {
                '__init__': lambda s: None,
                'inference': inf,
                'colors': self._inferences[key].colors,
                'directed': property(
                    lambda cls: cls.inference.directed
                ),
                'run': lambda cls, data, param: cls.inference.run(
                    data, 
                    param
                )
            }
        )
        return BenchmarkInf()

    def _generate_keys(self) -> Iterator[K]:
        for key in self._inferences.keys():
            if self.match(key):
                yield key
    
    def _save(self, path: Path) -> None:
        for inf_name, inf in self.items():
            output = (path / inf_name).with_suffix('.npz')
            np.savez_compressed(
                output,
                inference=np.array(dumps(inf))
            )

InferencesGenerator.register_defaults()

if __name__ == '__main__':
    print(InferencesGenerator.available_inferences())
    for name, inf in InferencesGenerator(verbose=True).items():
        print(name, inf, inf.colors)