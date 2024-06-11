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
    
    def _create_benchmark_inference(self, 
            name: str, 
            inference:Inference
        ) -> Inference:
        BenchmarkInf = type(
            f'{type(inference).__name__}',
            (type(inference),),
            {
                '__init__': lambda s: None,
                'inference': inference,
                'colors': self._inferences[name].colors,
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
    
    def _load_keys(self, path: Path) -> Iterator[K]:
        paths = self.match_rec(path)
        for p in paths:
            key = str(
                p
                .relative_to(path / self.sub_directory_name)
                .with_suffix('')
            )
            yield key
    
    def _load(self, path: Path) -> Iterator[Tuple[K, V]]:
        paths = self.match_rec(path)
        with alive_bar(
            len(paths),
            title='Loading inferences info',
            disable=not self.verbose
        ) as bar:
            for p in paths:
                bar.text(f'{p.absolute()}')
                name = str(
                    p
                    .relative_to(path / self.sub_directory_name)
                    .with_suffix('')
                )
                with np.load(p) as data:
                    inf = loads(data['inference'].item())
                    if not isinstance(inf, Inference):
                        raise RuntimeError(
                            f'{inf} is not an Inference object.'
                        )
                
                bar()
                yield name, inf

    def _load_values(self, path: Path) -> Iterator[V]:
        paths = self.match_rec(path)
        with alive_bar(
            len(paths),
            title='Loading inferences info',
            disable=not self.verbose
        ) as bar:
            for p in paths:
                bar.text(f'{p.absolute()}')
                with np.load(p) as data:
                    inf = loads(data['inference'].item())
                    if not isinstance(inf, Inference):
                        raise RuntimeError(
                            f'{inf} is not an Inference object.'
                        )
                
                bar()
                yield inf

    def _generate_keys(self) -> Iterator[K]:
        for key in self._inferences.keys():
            if self.match(key):
                yield key

    def _generate(self) -> Iterator[Tuple[K, V]]:
        inferences = {
            k:i for k,i in self._inferences.items()
            if self.match(k)
        }
        with alive_bar(
            len(inferences), 
            title='Generating inferences',
            disable=not self.verbose
        ) as bar:
            for name, inf_info in inferences.items():
                bar.text(name)
                if isinstance(inf_info.inference, Inference):
                    inf = inf_info.inference
                else:
                    inf = inf_info.inference()
                
                if not isinstance(inf, Inference):
                    raise RuntimeError(
                        (f'{inf_info.inference} is not an Inference subclass.')
                    )
                benchmark_inf = self._create_benchmark_inference(name, inf)
                bar()
                yield name, benchmark_inf

    def _generate_values(self) -> Iterator[V]:
        inferences = {
            k:i for k,i in self._inferences.items()
            if self.match(k)
        }
        with alive_bar(
            len(inferences), 
            title='Generating inferences',
            disable=not self.verbose
        ) as bar:
            for name, inf_info in inferences.items():
                bar.text(name)
                if isinstance(inf_info.inference, Inference):
                    inf = inf_info.inference
                else:
                    inf = inf_info.inference()
                
                if not isinstance(inf, Inference):
                    raise RuntimeError(
                        (f'{inf_info.inference} is not an Inference subclass.')
                    )
                benchmark_inf = self._create_benchmark_inference(name, inf)
                bar()
                yield benchmark_inf
    
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
    for name, inf in InferencesGenerator():
        print(name, inf, inf.colors)