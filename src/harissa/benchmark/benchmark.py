from typing import (
    List,
    Tuple,
    Union, 
    Optional,
    TypeAlias
)
from collections.abc import Iterator

from pathlib import Path
from time import perf_counter

import numpy as np

from harissa.core import NetworkModel, NetworkParameter, Inference, Dataset
from harissa.benchmark.generators import (
    GenericGenerator,
    DatasetsGenerator,
    InferencesGenerator
)

from harissa.plot.plot_benchmark import plot_benchmark

K : TypeAlias = Tuple[str, str, str, str]
V : TypeAlias = Tuple[
    NetworkParameter, 
    Inference, 
    Dataset, 
    Inference.Result, 
    float
]
class Benchmark(GenericGenerator[K, V]):
    def __init__(self,
        n_scores: int = 1, 
        path: Optional[Union[str, Path]] = None,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        verbose: bool = True
    ) -> None:
        self._generators = [
           DatasetsGenerator(),
           InferencesGenerator() 
        ]
        self.model = NetworkModel()
        self.n_scores = n_scores
        super().__init__('scores', include, exclude, path, verbose)

    # Aliases
    @property
    def datasets(self):
        return self._generators[0]
    
    @property
    def inferences(self):
        return self._generators[1]
    
    @property
    def networks(self):
        return self.datasets.networks
    
    def set_path(self, path: Path):
        super().set_path(path)
        self.datasets.path = path
        self.inferences.path = path

    def set_verbose(self, verbose: bool):
        super().set_verbose(verbose)
        self.datasets.verbose = verbose
        self.inferences.verbose = verbose

    def set_include(self, include):
        super().set_include(include)
        for _ in self.keys():
            pass
    
    def set_exclude(self, exclude):
        super().set_exclude(exclude)
        for _ in self.keys():
            pass
    
    def _load_value(self, path: Path, key: K) -> V:
        network , dataset = self.datasets[key[0], key[2]]
        inf = self.inferences[key[1]]

        result_path = path.joinpath(self.sub_directory_name, *key)
        result = inf[0].Result.load(
            result_path / 'result.npz', 
            load_extra=True
        )
        runtime = np.load(result_path / 'runtime.npy')

        return network, inf, dataset, result, runtime

    def _load_keys(self, path: Path) -> Iterator[K]:
        datasets_included = []
        inferences_included = []

        root = path / self.sub_directory_name

        for d_key in self.datasets.keys():
            for inf_name in self.inferences.keys():
                r_dir = root.joinpath(d_key[0], inf_name, d_key[1])
                for r_path in r_dir.iterdir():
                    key = (d_key[0], inf_name, d_key[1], r_path.stem)
                    if self.match(key):
                        if d_key not in datasets_included:
                            datasets_included.append(d_key)
                        if inf_name not in inferences_included:
                            inferences_included.append(inf_name)
                        
                        yield key

        self.datasets.include = datasets_included
        self.inferences.include = inferences_included

    def _generate_value(self, key: K) -> V:
        network, dataset = self.datasets[key[0], key[2]]
        inf = self.inferences[key[1]]
        
        self.model.parameter = network
        self.model.inference = inf[0]
        
        start = perf_counter()
        result = self.model.fit(dataset)
        runtime = perf_counter() - start
            
        return network, inf, dataset, result, runtime

    def _generate_keys(self) -> Iterator[K]:
        datasets_included = []
        inferences_included = []

        for dataset_key in self.datasets.keys():
            for inf_name in self.inferences.keys():
                for i in range(self.n_scores):
                    key = (dataset_key[0], inf_name, dataset_key[1], f'r{i+1}')
                    if self.match(key):
                        if dataset_key not in datasets_included:
                            datasets_included.append(dataset_key)
                        if inf_name not in inferences_included:
                            inferences_included.append(inf_name)
                        
                        yield key

        self.datasets.include = datasets_included
        self.inferences.include = inferences_included

    def _pre_load(self, path: Path):
        for generator in self._generators:
            generator.path = path

    def _post_load(self):
        for generator in self._generators:
            generator.path = self.path

    def _pre_generate(self):
        for generator in self._generators:
            generator.verbose = False

    def _post_generate(self):
        for generator in self._generators:
            generator.verbose = self.verbose

    def _pre_save(self, path):
        for generator in self._generators:
            generator.save(path)
            if generator.path is None:
                generator.path = path
            generator.verbose = False
            
    def _post_save(self):
        for generator in self._generators:
            generator.verbose = self.verbose

    def _save(self, path: Path) -> None:
        for (n, i, d, r), (*_, result, runtime) in self.items():
            output = path.joinpath(n, i, d, r)
            output.mkdir(parents=True, exist_ok=True)
            result.save(output / 'result', True)
            np.save(output / 'runtime.npy', np.array([runtime]))

        if self.path is None:
            self.path = path.parent

        self.save_reports(path.parent / 'reports')

    def reports(self, show_networks=False):
        return plot_benchmark(self, show_networks)

    def save_reports(self,
        path: Union[str, Path], 
        show_networks=False
    ) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        for fname, fig in zip(
            [
                'general.pdf', 
                'directed.pdf', 
                'undirected.pdf'
            ], 
             self.reports(show_networks)
        ):
            fig.savefig(path / fname)
        return path
    

if __name__ == '__main__':
    benchmark = Benchmark()
    benchmark.datasets.path = 'test_benchmark'
    benchmark.datasets.include = [('BN8', '*')]
    benchmark.save('test_benchmark')
    
    benchmark = Benchmark()
    benchmark.path='test_benchmark'
    benchmark.networks.exclude = ['Trees*']
    print(benchmark.save('test_benchmark2'))
    print(benchmark.save_reports('test_reports', True))
    
