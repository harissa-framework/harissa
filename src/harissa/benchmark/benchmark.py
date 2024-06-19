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
        n_run: int = 1, 
        path: Optional[Union[str, Path]] = None,
        include: List[str] = [('*', '*', '*', '*')],
        exclude: List[str] = [],
        verbose: bool = True
    ) -> None:
        super().__init__('scores', include, exclude, path, verbose)
        self._generators = [
           DatasetsGenerator(path=path),
           InferencesGenerator(path=path) 
        ]
        self._old_generators_path, self._old_generators_verbose = (
            [gen.path for gen in self._generators],
            [gen.verbose for gen in self._generators]
        ) 
        self._model = NetworkModel()
        self.n_run = n_run

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
        root = path / self.sub_directory_name

        for d_key in self.datasets.keys():
            for inf_name in self.inferences.keys():
                r_dir = root.joinpath(d_key[0], inf_name, d_key[1])
                for r_path in r_dir.iterdir():
                    key = (d_key[0], inf_name, d_key[1], r_path.stem)
                    if self.match(key):
                        yield key

    def _generate_value(self, key: K) -> V:
        network, dataset = self.datasets[key[0], key[2]]
        inf = self.inferences[key[1]]
        
        self._model.parameter = network
        self._model.inference = inf[0]
        
        start = perf_counter()
        result = self._model.fit(dataset)
        runtime = perf_counter() - start
            
        return network, inf, dataset, result, runtime

    def _generate_keys(self) -> Iterator[K]:
        for dataset_key in self.datasets.keys():
            for inf_name in self.inferences.keys():
                for i in range(self.n_run):
                    key = (dataset_key[0], inf_name, dataset_key[1], f'r{i+1}')
                    if self.match(key):
                        yield key

    def _pre_load(self, path: Path):
        for i, generator in enumerate(self._generators):
            self._old_generators_path[i] = generator.path
        
        for generator in self._generators:
            if self.path == generator.path:
                generator.path = path

    def _post_load(self):
        for i, path in enumerate(self._old_generators_path):
            self._generators[i].path = path

    def _pre_generate(self):
        for i, generator in enumerate(self._generators):
            self._old_generators_verbose[i] = generator.verbose
        
        for generator in self._generators:
            generator.verbose = False

    def _post_generate(self):
        for i, verbose in enumerate(self._old_generators_verbose):
            self._generators[i].verbose = verbose

    def _pre_save(self, path):
        for i, generator in enumerate(self._generators):
            self._old_generators_path[i] = generator.path
            self._old_generators_verbose[i] = generator.verbose

        for generator in self._generators:
            generator.verbose = self.verbose
            generator.save(path)
            if generator.path is None:
                generator.path = path
            generator.verbose = False
            
    def _post_save(self):
        for i, path in enumerate(self._old_generators_path):
            self._generators[i].path = path

        for i, verbose in enumerate(self._old_generators_verbose):
            self._generators[i].verbose = verbose
        
    def _save(self, path: Path) -> None:
        for (n, i, d, r), (*_, result, runtime) in self.items():
            output = path.joinpath(n, i, d, r)
            output.mkdir(parents=True, exist_ok=True)
            result.save(output / 'result', True)
            np.save(output / 'runtime.npy', np.array([runtime]))

        old_path = self.path
        self.path = path.parent

        self.save_reports(path.parent / 'reports')

        self.path = old_path

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
    benchmark.networks.include = ['BN8']
    benchmark.save('test_benchmark')
    
    benchmark = Benchmark(path='test_benchmark')
    benchmark.networks.exclude = ['Trees*']
    print(benchmark.save('test_benchmark2'))
    print(benchmark.save_reports('test_reports', True))
    
