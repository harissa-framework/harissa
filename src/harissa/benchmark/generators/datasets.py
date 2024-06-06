from typing import (
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
from harissa.utils.progress_bar import alive_bar

from harissa.core import Dataset, NetworkModel
from harissa.simulation import BurstyPDMP
from harissa.benchmark.generators.generic import GenericGenerator
from harissa.benchmark.generators.networks import NetworksGenerator

K: TypeAlias = Tuple[str, int]
V: TypeAlias = Dataset
class DatasetsGenerator(GenericGenerator[K, V]):
    def __init__(self, 
        networks: Optional[NetworksGenerator] = None,
        time_points : npt.NDArray[np.float_] = np.array([
            0, 6, 12, 24, 36, 48, 60, 72, 84, 96
        ], dtype=float),
        n_cells: int = 100,
        burn_in_duration: float = 5.0,
        n_datasets : int = 10,
        include: Optional[List[str]] = None, 
        exclude: Optional[List[str]] = None,
        path: Optional[Union[str, Path]] = None,
        verbose: bool = False
    ) -> None:
        super().__init__('datasets', include, exclude, path, verbose)
            
        self.networks = (
            networks or NetworksGenerator()
        )
    
        self.model = NetworkModel(
            simulation=BurstyPDMP(use_numba=True)
        )
        self.simulate_dataset_parameters = {
            'time_points': time_points, 
            'n_cells' : n_cells,
            'burn_in_duration': burn_in_duration
        }
        self.n_datasets = n_datasets

    def _keys(self, path: Path | None) -> Iterator[K]:
        if path is not None:
            for p in self.match_rec(path):
                network_name = str(
                    p
                    .relative_to(path / self.sub_directory_name)
                    .with_suffix('')
                )
                yield network_name, int(p.stem)
        else:
            for network_name in self.networks.keys():
                if isinstance(self.n_datasets, int):
                    n_datasets = self.n_datasets    
                else: 
                    n_datasets = self.n_datasets.get(network_name, 10)
                 
                for i in range(n_datasets):
                    if self.match(Path(network_name) / str(i+1)):
                        yield network_name, i+1


    
    def _load(self, path: Path) -> Iterator[Tuple[K, V]]:
        paths = self.match_rec(path)
        self.networks.path = path.parent
        networks_included = []
        with alive_bar(
            len(paths),
            title='Loading datasets',
            disable=not self.verbose
        ) as bar:
            for p in paths:
                bar.text(f'{p.absolute()}')
                network_name = str(
                    p.parent
                    .relative_to(path / self.sub_directory_name)
                    .with_suffix('')
                )
                if network_name not in networks_included:
                    networks_included.append(network_name)

                dataset = Dataset.load(p)
                bar()
                yield (network_name, int(p.stem)), dataset
        
        self.networks.include = networks_included
        self.remove_tmp_dir(path)

    def _generate(self) -> Iterator[Tuple[K, V]]:
        datasets_per_network = {}
        for k, i in self.keys():
            if k not in datasets_per_network:
                datasets_per_network[k] = [i]
            else:
                datasets_per_network[k].append(i)
        self.networks.include = list(datasets_per_network.keys())

        with alive_bar(
            int(np.sum([len(d) for d in datasets_per_network.values()])),
            title='Generating datasets',
            disable=not self.verbose
        ) as bar:
            for network_name, network in self.networks:
                self.model.parameter = network
                for i in datasets_per_network[network_name]:
                    bar.text(f'Generating {network_name} - dataset {i}')
                    dataset = self.model.simulate_dataset(
                        **self.simulate_dataset_parameters
                    )
                    bar()
                    yield (network_name, i), dataset
    
    def _save(self, path: Path) -> None:
        self.networks.save(path.parent)
        self.networks.path = path.parent
        for (name, i) , dataset in self:
            output = path / name / f'{i}.npz'
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
        exclude=['FN4/3', 'FN4/7'],
        path='test_datasets',
        verbose= True
    )
    gen.save('test_datasets2')