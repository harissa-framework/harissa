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
import numpy.typing as npt

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
        n_datasets : Union[int, Dict[str, int]] = default_n_datasets,
        include: List[K] = [('*', '*')], 
        exclude: List[K] = [],
        path: Optional[Union[str, Path]] = None,
        verbose: bool = False
    ) -> None:
        super().__init__('datasets', include, exclude, path, verbose)
        self.n_datasets = n_datasets
        self.simulate_parameters = simulate_parameters
        self._model = NetworkModel(
            simulation=BurstyPDMP(use_numba=True)
        )

        self.networks = NetworksGenerator(path=path)
        self._old_network_path, self._old_network_verbose = (
            self.networks.path, 
            self.networks.verbose
        )

    @property
    def networks(self):
        return self._networks
    
    @networks.setter
    def networks(self, network_gen):
        if not isinstance(network_gen, NetworksGenerator):
            raise TypeError(f'{network_gen} must be a NetworksGenerator.')
        
        self._networks = network_gen
    
    def _load_value(self, path: Path, key: K) -> V:
        network = self.networks[key[0]]
        dataset = Dataset.load(
            path / self.sub_directory_name / key[0] / f'{key[1]}.npz' 
        )
        return network, dataset

    def _load_keys(self, path: Path) -> Iterator[K]:
        for network_name in self.networks.keys():
            dataset_dir = path / self.sub_directory_name / network_name

            for dataset_path in dataset_dir.iterdir():
                key = (network_name, dataset_path.stem)
                if self.match(key):
                    yield key

    def _generate_value(self, key: K) -> V:
        network = self.networks[key[0]]
        self._model.parameter = network
        
        dataset = self._model.simulate_dataset(
            **self.simulate_parameters[key]
        )
        
        return network, dataset

    def _generate_keys(self) -> Iterator[K]:
        n_datasets = {}
        parameters = {}
        for network_name in self.networks.keys():
            if isinstance(self.n_datasets, int):
                n = self.n_datasets    
            else: 
                n = self.n_datasets.get(
                    network_name, 
                    default_n_datasets
                )
            for i in range(n):
                key = (network_name, f'd{i+1}')
                if self.match(key):
                    if network_name not in n_datasets:
                        n_datasets[network_name] = n

                    if key in self.simulate_parameters:
                        parameters[key] = self.simulate_parameters[key]
                    elif key[0] in self.simulate_parameters:
                        parameters[key] = self.simulate_parameters[key[0]]
                    else:
                        parameters[key] = {
                            k:self.simulate_parameters.get(k, v)
                            for k, v in default_simulate_parameters.items()
                        }

                    yield key

        self.n_datasets = n_datasets
        self.simulate_parameters = parameters

    def _pre_load(self, path: Path):
        self._old_network_path = self.networks.path
        if self.path == self.networks.path: 
            self.networks.path = path

    def _post_load(self):
        self.networks.path = self._old_network_path

    def _pre_generate(self):
        self._old_network_verbose = self.networks.verbose
        self.networks.verbose = False
    
    def _post_generate(self):
        self.networks.verbose = self._old_network_verbose

    def _pre_save(self, path: Path):
        self._old_network_verbose = self.networks.verbose
        self.networks.verbose = self.verbose
        
        self.networks.save(path)
        self._old_network_path = self.networks.path
        
        self.networks.verbose = False
        if self.networks.path is None:
            self.networks.path = path
    
    def _post_save(self):
        self.networks.verbose = self._old_network_verbose
        self.networks.path = self._old_network_path
    
    def _save(self, path: Path) -> None:
        for (network_name, dataset_name) , (_, dataset) in self.items():
            output = path / network_name / f'{dataset_name}.npz'
            output.parent.mkdir(parents=True, exist_ok=True)
            dataset.save(output)
    
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