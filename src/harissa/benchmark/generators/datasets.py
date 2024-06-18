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

class DatasetsGenerator(GenericGenerator[K, V]):
    def __init__(self,
        time_points : npt.NDArray[np.float_] = np.array([
            0, 6, 12, 24, 36, 48, 60, 72, 84, 96
        ], dtype=float),
        n_cells: int = 100,
        burn_in_duration: float = 5.0,
        n_datasets : Union[int, Dict[str, int]] = 10,
        include: Optional[List[str]] = None, 
        exclude: Optional[List[str]] = None,
        path: Optional[Union[str, Path]] = None,
        verbose: bool = False
    ) -> None:
        self.networks = NetworksGenerator()
    
        self.model = NetworkModel(
            simulation=BurstyPDMP(use_numba=True)
        )
        self.simulate_dataset_parameters = {
            'time_points': time_points, 
            'n_cells' : n_cells,
            'burn_in_duration': burn_in_duration
        }
        self.n_datasets = n_datasets
        super().__init__('datasets', include, exclude, path, verbose)


    def set_path(self, path: Path):
        super().set_path(path)
        self.networks.path = path

    def set_verbose(self, verbose: bool):
        super().set_verbose(verbose)
        self.networks.verbose = verbose

    def set_include(self, include):
        super().set_include(include)
        for _ in self.keys():
            pass

    def set_exclude(self, exclude):
        super().set_exclude(exclude)
        for _ in self.keys():
            pass

    def _load_value(self, path: Path, key: K) -> V:
        network = self.networks[key[0]]
        dataset = Dataset.load(
            path / self.sub_directory_name / key[0] / f'{key[1]}.npz' 
        )
        return network, dataset

    def _load_keys(self, path: Path) -> Iterator[K]:
        network_included = []
        for network_name in self.networks.keys():
            dataset_dir = path / self.sub_directory_name / network_name
            for dataset_path in dataset_dir.iterdir():
                dataset_name = dataset_path.stem
                if self.match((network_name, dataset_name)):
                    if network_name not in network_included:
                        network_included.append(network_name)
                    yield network_name, dataset_name
    
        self.networks.include = network_included

    def _generate_value(self, key: K) -> V:
        network = self.networks[key[0]]
        self.model.parameter = network
        
        dataset = self.model.simulate_dataset(
            **self.simulate_dataset_parameters
        )
        
        return network, dataset

    def _generate_keys(self) -> Iterator[K]:
        network_included = []
        for network_name in self.networks.keys():
            if isinstance(self.n_datasets, int):
                n_datasets = self.n_datasets    
            else: 
                n_datasets = self.n_datasets.get(network_name, 10)
                
            for i in range(n_datasets):
                dataset_name = f'd{i+1}'
                if self.match((network_name, dataset_name)):
                    if network_name not in network_included:
                        network_included.append(network_name)
                    yield network_name, dataset_name

        self.networks.include = network_included

    def _pre_load(self, path: Path):
        self.networks.path = path

    def _post_load(self):
        self.networks.path = self.path

    def _pre_generate(self):
        self.networks.verbose = False
    
    def _post_generate(self):
        self.networks.verbose = self.verbose


    def _pre_save(self, path: Path):
        self.networks.save(path)
        self.networks.verbose = False
        if self.path is None:
            self.networks.path = path
    
    def _post_save(self):
        self.networks.verbose = self.verbose
        if self.path is None:
            self.networks.path = None
    
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