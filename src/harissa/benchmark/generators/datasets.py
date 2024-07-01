from typing import (
    Dict,
    Tuple, 
    List, 
    Union, 
    Optional,
    TypeAlias
)

from collections.abc import Iterator

from pathlib import Path
import numpy as np

from harissa.core import Dataset, NetworkModel, NetworkParameter
from harissa.simulation import BurstyPDMP
from harissa.benchmark.generators.generic import GenericGenerator
from harissa.benchmark.generators.networks import NetworksGenerator

K: TypeAlias = Tuple[str, str]
V: TypeAlias = Tuple[NetworkParameter, Dataset]

default_simulate_parameters: Dict = {
    'time_points': np.array([
        0, 6, 12, 24, 36, 48, 60, 72, 84, 96
    ], dtype=float),
    'n_cells': 100,
    'burn_in_duration': 5.0
}

default_n_datasets: int = 10


N_DatasetsType: TypeAlias = Union[
    Union[int, List[str]], 
    Dict[str, Union[int, List[str]]]
]

class DatasetsGenerator(GenericGenerator[K, V]):
    """
    Generator of datasets
    """
    def __init__(self,
        simulate_parameters: Dict = default_simulate_parameters,
        n_datasets: N_DatasetsType = default_n_datasets,
        include: List[K] = [('*', '*')], 
        exclude: List[K] = [],
        path: Optional[Union[str, Path]] = None,
        verbose: bool = False
    ) -> None:
        self.networks = NetworksGenerator(verbose=verbose)
        
        super().__init__('datasets', include, exclude, path, verbose)
        
        self.n_datasets = n_datasets
        self.simulate_parameters = simulate_parameters
        self._model = NetworkModel(
            simulation=BurstyPDMP(use_numba=True)
        )

    def _set_path(self, path: Path):
        """
        Set the path to self and to the networks generator.
        """
        super()._set_path(path)
        self.networks.path = path

    @property
    def networks(self):
        """
        Networks generator
        """
        return self._networks
    
    @networks.setter
    def networks(self, network_gen):
        if not isinstance(network_gen, NetworksGenerator):
            raise TypeError(f'{network_gen} must be a NetworksGenerator.')
        
        self._networks = network_gen
    
    def _load_value(self, key: K) -> V:
        """
        Load a value from a key.

        Parameters
        ----------
        key : 
            input key

        Raises
        ------
        KeyError
        """
        network = self.networks[key[0]]
        path = self._to_path(key).with_suffix('.npz')
        
        if not path.exists():
            raise KeyError(f'{key} is invalid. {path} does not exist.')
         
        dataset = Dataset.load(path)
        
        return network, dataset

    def _load_keys(self) -> Iterator[K]:
        """
        Load all the keys

        Yields
        ------
        K
        """
        for network_name in self.networks.keys():
            dataset_dir = self._to_path(network_name)
            for dataset_path in dataset_dir.iterdir():
                key = (network_name, dataset_path.stem)
                if self.match(key):
                    yield key

    def _get_n_datasets(self, network_key: str) -> List[str]:
        """
        Get the dataset names given a network name.

        Parameters
        ----------
        network_key
            input network key

        """
        if isinstance(self.n_datasets, dict):
            n_datasets = self.n_datasets.get(network_key, default_n_datasets)
        else:
            n_datasets = self.n_datasets

        if isinstance(n_datasets, int):
            n_datasets = [f'd{i+1}' for i in range(n_datasets)]
    
        return n_datasets

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
        n_datasets = self._get_n_datasets(key[0])
        if key[1] not in n_datasets:
            raise KeyError(
                f'{key} is invalid. {key[1]} must be inside {n_datasets}.'
            )

        network = self.networks[key[0]]
        self._model.parameter = network

        if key in self.simulate_parameters:
            parameters = self.simulate_parameters[key]
        elif key[0] in self.simulate_parameters:
            parameters = self.simulate_parameters[key[0]]
        else:
            parameters = {
                k:self.simulate_parameters.get(k, v)
                for k, v in default_simulate_parameters.items()
            }
        
        dataset = self._model.simulate_dataset(
            **parameters
        )
        
        return network, dataset

    def _generate_keys(self) -> Iterator[K]:
        """
        Generate all the keys

        Yields
        ------
        K
        """
        for network_name in self.networks.keys():
            for dataset_name in self._get_n_datasets(network_name):
                key = (network_name, dataset_name)
                if self.match(key):
                    yield key

    def _save_item(self, path: Path, item: Tuple[K, V]):
        """
        Save an item

        Parameters
        ----------
        path
            path where to save
        item : 
            item to save

        """
        key, (network, dataset) = item
        output = self._to_path(key, path).with_suffix('.npz')
        output.parent.mkdir(parents=True, exist_ok=True)

        dataset.save(output)
        self.networks.save_item(path, key[0], network)

    
if __name__ == '__main__':
    n_datasets = {'BN8': 2, 'CN5': 5, 'FN4': 10, 'FN8': 1}
    gen = DatasetsGenerator(
        n_datasets=n_datasets, 
        verbose=True
    )
    gen.networks.include = list(n_datasets.keys())
    gen.save('test_datasets')

    gen = DatasetsGenerator(
        exclude=[('FN4','d3'), ('FN4', 'd7')],
        path='test_datasets',
        verbose= True
    )
    gen.save('test_datasets2')