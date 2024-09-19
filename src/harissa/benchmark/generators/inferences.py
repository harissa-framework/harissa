from typing import (
    Dict,
    List,
    Tuple,
    Callable,
    Union,
    Optional
)
from collections.abc import Iterator

from pathlib import Path
from json import load, dump

import matplotlib.colors
import matplotlib.pyplot
import numpy as np
import numpy.typing as npt
import matplotlib

from harissa.benchmark.generators.generic import GenericGenerator
from harissa.core import Inference
from harissa.inference import Hartree, Cardamom, Pearson

K = str
V = Tuple[Inference, npt.NDArray[np.float64]]
class InferencesGenerator(GenericGenerator[K, V]):
    """
    Generator of inference methods

    """
    _inferences: Dict[
        str,
        Tuple[
            Union[Inference, Callable[[], Inference]],
            npt.NDArray[np.float64],
            Dict
        ]
    ] = {}
    color_map: matplotlib.colors.Colormap = matplotlib.pyplot.get_cmap('tab20')

    def __init__(self,
        include: List[str] = ['*'],
        exclude: List[str] = [],
        path: Optional[Union[str, Path]] = None,
        verbose: bool = False
    ) -> None:
        super().__init__('inferences', include, exclude, path, verbose)

    @classmethod
    def register(cls,
        name: str,
        inference: Union[Inference, Callable[[], Inference]],
        colors: npt.NDArray[np.float64],
        inference_kwargs: Dict = {}
    ) -> None:
        """
        Register inferences instances or function that creates inferences,
        later used during the generation.

        Parameters
        ----------
        name
            inference method name
        inference
            inference or function to be registered
        colors
            2D array representing RGBA colors (foreground and background) used
            during plots.
        inference_kwargs
            keywords arguments used by the function to generate an inference

        Raises
        ------
        TypeError
            If the inference method does not have the right type.
        ValueError
            If the name is already taken
        """
        if name not in cls._inferences:
            if isinstance(inference, (Inference, Callable)):
                cls._inferences[name] = (inference, colors, inference_kwargs)
            else:
                raise TypeError(('inference_callable must be a callable '
                             'that returns a Inference sub class.'))
        else:
            raise ValueError((f'{name} is already taken. '
                              f'Cannot register {inference}.'))

    @classmethod
    def register_defaults(cls) -> None:
        """
        Register the default inference methods.
        """
        cls.register(
            'Hartree',
            Hartree,
            np.array([cls.color_map(6), cls.color_map(7)])
        )
        cls.register(
            'Cardamom',
            Cardamom,
            np.array([cls.color_map(8), cls.color_map(9)])
        )
        cls.register(
            'Pearson',
            Pearson,
            np.array([cls.color_map(14), cls.color_map(15)])
        )

    @classmethod
    def unregister_all(cls) -> None:
        """
        Clear the registered inference methods
        """
        cls._networks = {}

    @classmethod
    def available_inferences(cls) -> List[str]:
        """
        Returns a list of registered inference method names

        """
        return list(cls._inferences.keys())

    def _load_value(self, key: K) -> V:
        """
        Load a value from a key.

        Parameters
        ----------
        key:
            input key

        Raises
        ------
        KeyError
        """
        path = self._to_path(key).with_suffix('.json')

        if not path.exists():
            raise KeyError(f'{key} is invalid. {path} does not exist.')

        inf = Inference.load_json(path)
        if not isinstance(inf, Inference):
            raise RuntimeError(
                f'{inf} is not an Inference object.'
            )
        with open(path, 'r') as fp:
            colors = np.array(load(fp)['benchmark_colors'])

        return inf, colors

    def _generate_value(self, key: K) -> V:
        """
        Generate a value from a key

        Parameters
        ----------
        key
            input key

        Raises
        ------
        KeyError
        """
        if key not in self._inferences:
            raise KeyError(f'{key} is invalid. {key} is not registered.')

        inference, colors, kwargs = self._inferences[key]
        if isinstance(inference, Inference):
            inf = inference
        else:
            inf = inference(**kwargs)

        if not isinstance(inf, Inference):
            raise RuntimeError(
                (f'{inference} is not an Inference subclass.')
            )

        return inf, colors

    def _generate_keys(self) -> Iterator[K]:
        """
        Generate all the keys

        Yields
        ------
        K
        """
        for key in self._inferences.keys():
            if self.match(key):
                yield key

    def _save_item(self, path: Path, item: Tuple[K, V]):
        """
        Save an item

        Parameters
        ----------
        path
            path where to save
        item
            item to save

        """
        key, (inf, colors) = item
        output = self._to_path(key, path).with_suffix('.json')
        output.parent.mkdir(parents=True, exist_ok=True)

        inf.save_json(output)
        with open(output, 'r') as fp:
            inf_dict = load(fp)
        # inf_dict['benchmark_key'] = key
        inf_dict['benchmark_colors'] = colors.tolist()

        with open(output, 'w') as fp:
            dump(inf_dict, fp, indent=4)

InferencesGenerator.register_defaults()

if __name__ == '__main__':
    print(InferencesGenerator.available_inferences())
    gen = InferencesGenerator(verbose=True)
    for name, (inf, colors) in gen.items():
        print(name)
        print(inf)
        print(colors)

    gen.save('test_inferences')
    gen.path = 'test_inferences'

    for name, (inf, colors) in gen.items():
        print(name)
        print(inf)
        print(colors)
