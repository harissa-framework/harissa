from typing import (
    Dict, 
    List,
    Tuple, 
    Callable, 
    Union, 
    Optional
)

from dataclasses import dataclass
from pathlib import Path
from alive_progress import alive_bar
from dill import loads, dumps

import matplotlib.colors
import matplotlib.pyplot
import numpy as np
import numpy.typing as npt
import matplotlib

from harissa.benchmark.generators.generic import GenericGenerator
from harissa.core import Inference
from harissa.inference import Hartree, Cardamom, Pearson


@dataclass
class InferenceInfo:
    inference: Union[Inference, Callable[[], Inference]]
    colors: npt.NDArray

class InferencesGenerator(GenericGenerator[Tuple[Inference, np.array]]):
    _inferences : Dict[str, InferenceInfo] = {}
    color_map: matplotlib.colors.Colormap = matplotlib.pyplot.get_cmap('tab20')

    def __init__(self,
        include: Optional[List[str]] = None, 
        exclude: Optional[List[str]] = None,
        path: Optional[Union[str, Path]] = None
    ) -> None:
        super().__init__('inferences', include, exclude, path)

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
    
    def _load(self, path: Path) -> None:
        self._items = {}
        paths = self.match_rec(path)
        with alive_bar(
            len(paths),
            title='Loading inferences info'
        ) as bar:
            for p in paths:
                bar.text(f'{p.absolute()}')
                name = str(p.relative_to(path).with_suffix(''))
                with np.load(p) as data:

                    info = InferenceInfo(
                        loads(data['inference'].item()),
                        data['colors']
                    )
                    
                    if isinstance(info.inference, Inference):
                        if name not in self._inferences:
                            self.register(name, info)
                        self._items[name] = (info.inference, info.colors)
                    else:
                        raise RuntimeError(
                            f'{info.inference} is not an Inference object.'
                        )
                
                bar()


    def _generate(self) -> None:
        self._items = {}
        inferences = {
            k:i for k,i in self._inferences.items()
            if self.match(k)
        }
        with alive_bar(len(inferences), title='Generating inferences') as bar:
            for name, inf_info in inferences.items():
                bar.text(f'{name}')
                if isinstance(inf_info.inference, Inference):
                    inf = inf_info.inference
                else:
                    inf = inf_info.inference()
                
                if isinstance(inf, Inference):
                    self._items[name] = (inf, self._inferences[name].colors)
                else:
                    raise RuntimeError(
                        (f'{inf_info.inference} is not a callable'
                        ' that returns a Inference sub class.')
                    )
                bar()
    
    def _save(self, path: Path) -> None:
        with alive_bar(
            len(self.inferences), 
            title='Saving Inferences Info'
        ) as bar:
            for inf_name, (inf, colors) in self.inferences.items():
                output = (path / inf_name).with_suffix('.npz')
                bar.text(f'{output.absolute()}')
                np.savez_compressed(
                    output,
                    inference=np.array(dumps(inf)),
                    colors=colors
                )

InferencesGenerator.register_defaults()

if __name__ == '__main__':
    print(InferencesGenerator.available_inferences())