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

class DatasetsGenerator(GenericGenerator[K, V]):
    def __init__(self,
        simulate_parameters: Dict = default_simulate_parameters,
        n_datasets: Union[int, Dict[str, int]] = default_n_datasets,
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
        super()._set_path(path)
        self.networks.path = path

    @property
    def networks(self):
        return self._networks
    
    @networks.setter
    def networks(self, network_gen):
        if not isinstance(network_gen, NetworksGenerator):
            raise TypeError(f'{network_gen} must be a NetworksGenerator.')
        
        self._networks = network_gen
    
    def _load_value(self, key: K) -> V:
        network = self.networks[key[0]]
        path = self._to_path(key).with_suffix('.npz')
        dataset = Dataset.load(path)
        
        return network, dataset

    def _load_keys(self) -> Iterator[K]:
        for network_name in self.networks.keys():
            dataset_dir = self._to_path(network_name)
            for dataset_path in dataset_dir.iterdir():
                key = (network_name, dataset_path.stem)
                if self.match(key):
                    yield key

    def _generate_value(self, key: K) -> V:
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
        for network_name in self.networks.keys():
            if isinstance(self.n_datasets, int):
                n_datasets = self.n_datasets    
            else: 
                n_datasets = self.n_datasets.get(
                    network_name, 
                    default_n_datasets
                )
            for i in range(n_datasets):
                key = (network_name, f'd{i+1}')
                if self.match(key):
                    yield key

    def _save_item(self, path: Path, item: Tuple[K, V]):
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