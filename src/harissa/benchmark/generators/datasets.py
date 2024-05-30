from typing import (
    Dict, 
    List, 
    Union, 
    Optional
)

from pathlib import Path
import numpy as np
import numpy.typing as npt
from alive_progress import alive_bar

from harissa.core import Dataset, NetworkModel
from harissa.simulation import BurstyPDMP
from harissa.benchmark.generators.generic import GenericGenerator
from harissa.benchmark.generators.networks import NetworksGenerator

class DatasetsGenerator(GenericGenerator[npt.NDArray[Dataset]]):
    def __init__(self, 
        networks_generator: Optional[NetworksGenerator] = None,
        time_points : npt.NDArray[np.float_] = np.array([
            0, 6, 12, 24, 36, 48, 60, 72, 84, 96
        ], dtype=float),
        n_cells: int = 100,
        burn_in_duration: float = 5.0,
        n_datasets : int = 10,
        include: Optional[List[str]] = None, 
        exclude: Optional[List[str]] = None,
        path: Optional[Union[str, Path]] = None
    ) -> None:
        super().__init__('datasets', include, exclude, path)
            
        self.networks_generator = (
            networks_generator or NetworksGenerator(include, exclude, path)
        )
    
        self.model = NetworkModel(simulation=BurstyPDMP(
            use_numba=True
        ))
        self.simulate_dataset_parameters = {
            'time_points': time_points, 
            'n_cells' : n_cells,
            'burn_in_duration': burn_in_duration
        }
        self.n_datasets = n_datasets

    # Alias
    @property
    def datasets(self) -> Dict[str, npt.NDArray[Dataset]]:
        return self.items
    
    def _load(self, path: Path) -> None:
        self._items = {}
        paths = self.match_rec(path)
        with alive_bar(
            len(paths),
            title='Loading datasets'
        ) as bar:
            for p in paths:
                bar.text(f'{p.absolute()}')
                name = str(p.relative_to(path).with_suffix(''))
                
                with np.load(p) as data:
                    self._items[name] = np.array([
                        Dataset(
                            data[f'time_points_{i}'], 
                            data[f'count_matrix_{i}'],
                            data.get(f'gene_names_{i}', None)
                        )
                        for i in range(1, data['nb_datasets'].item() + 1)
                    ])
                bar()

    def _generate(self) -> None:
        self._items = {}
        with alive_bar(
            len(self.networks_generator.networks) * self.n_datasets,
            title='Generating datasets'
        ) as bar:
            for name, network in self.networks_generator.networks.items():
                self.model.parameter = network
                self._items[name] = np.empty(self.n_datasets, dtype=object)
                for i in range(self.n_datasets):
                    bar.text(f'Generating {name} - dataset {i+1}')
                    self._items[name][i] = self.model.simulate_dataset(
                        **self.simulate_dataset_parameters
                    )
                    bar()
    
    def _save(self, path: Path) -> None:
        self.networks_generator.save(path.parent)
        with alive_bar(len(self.datasets), title='Saving datasets') as bar:
            for name, datasets in self.datasets.items():
                output = path / name
                output.parent.mkdir(parents=True, exist_ok=True)

                bar.text(f'{output.absolute()}')
                datasets_dict = {'nb_datasets' : np.array(len(datasets))}
                for i, d in enumerate(datasets, 1):
                    datasets_dict[f'time_points_{i}'] = d.time_points
                    datasets_dict[f'count_matrix_{i}'] = d.count_matrix
                    if d.gene_names is not None:
                        datasets_dict[f'gene_names_{i}'] = d.gene_names

                np.savez_compressed(output, **datasets_dict)
                bar()