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
from dill import loads

import numpy as np

from alive_progress import alive_bar

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
        self.datasets = DatasetsGenerator()
        self.inferences = InferencesGenerator()
        self.model = NetworkModel()
        self.n_scores = n_scores
        super().__init__('scores', include, exclude, path, verbose)

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

    # Alias
    @property
    def networks(self):
        return self.datasets.networks

    def _load_keys(self, path: Path) -> Iterator[K]:
        self.datasets.path = path
        self.inferences.path = path

        yield from self._generate_keys()

        self.datasets.path = self.path
        self.inferences.path = self.path

        self.remove_tmp_dir(path)

    def _load(self, path: Path) -> Iterator[Tuple[K, V]]:
        scores_keys = list(self._load_keys(path))

        with alive_bar(len(scores_keys), title='Loading scores') as bar:
            for key in scores_keys:
                network = NetworkParameter.load(
                    path / self.networks.sub_directory_name / f'{key[0]}.npz'
                )

                with np.load(
                    path / self.inferences.sub_directory_name / f'{key[1]}.npz'
                ) as data:
                    inf = loads(data['inference'].item())

                dataset = Dataset.load(
                    path.joinpath(
                        self.datasets.sub_directory_name, 
                        key[0], 
                        f'{key[2]}.npz'
                    )
                )

                result_path = path.joinpath(self.sub_directory_name, *key)
                result = inf.Result.load(
                    result_path / 'result.npz', 
                    load_extra=True
                )
                runtime = np.load(result_path / 'runtime.npy')

                bar()
                yield key, (network, inf, dataset, result, runtime)
    
    def _load_values(self, path: Path) -> Iterator[V]:
        scores_keys = list(self._load_keys(path))

        with alive_bar(len(scores_keys), title='Loading scores') as bar:
            for key in scores_keys:
                network = NetworkParameter.load(
                    path / self.networks.sub_directory_name / f'{key[0]}.npz'
                )

                with np.load(
                    path / self.inferences.sub_directory_name / f'{key[1]}.npz'
                ) as data:
                    inf = loads(data['inference'].item())

                dataset = Dataset.load(
                    path.joinpath(
                        self.datasets.sub_directory_name, 
                        key[0], 
                        f'{key[2]}.npz'
                    )
                )

                result_path = path.joinpath(self.sub_directory_name, *key)
                result = inf.Result.load(
                    result_path / 'result.npz', 
                    load_extra=True
                )
                runtime = np.load(result_path / 'runtime.npy')

                bar()
                yield network, inf, dataset, result, runtime

    def _generate_keys(self) -> Iterator[K]:
        datasets_included = []
        inferences_included = []

        for network_name, data_name in self.datasets.keys():
            for inf_name in self.inferences.keys():
                for i in range(self.n_scores):
                    key = (network_name, inf_name, data_name, f'r{i+1}')
                    if self.match(Path().joinpath(*key)):
                        dataset_key = str(Path(network_name) / data_name)
                        if dataset_key not in datasets_included:
                            datasets_included.append(dataset_key)
                        if inf_name not in inferences_included:
                            inferences_included.append(inf_name)
                        
                        yield key

        self.datasets.include = datasets_included
        self.inferences.include = inferences_included

    def _generate(self) -> Iterator[K, V]:
        self.datasets.verbose = False
        self.inferences.verbose = False

        with alive_bar(
            len(self),
            title='Generating scores',
            disable=not self.verbose
        ) as bar:
            for (n_name, d_name), (network, dataset) in self.datasets.items():
                self.model.parameter = network
                for inf_name, inf in self.inferences:
                    self.model.inference = inf
                    for i in range(self.n_scores):
                        key = (n_name, inf_name, d_name, f'r{i+1}')
                        key_str = '-'.join(key[:-1])
                        if self.n_scores > 1:
                            key_str += f'-{key[-1]}'
                        
                        bar.text(f'Score {key_str}')
                        start = perf_counter()
                        result = self.model.fit(dataset)
                        runtime = perf_counter() - start
                        bar()
                        yield key, (network, inf, dataset, result, runtime)
        
        self.datasets.verbose = self.verbose
        self.inferences.verbose = self.verbose

    def _generate_values(self) -> Iterator[V]:   
        self.datasets.verbose = False
        self.inferences.verbose = False

        with alive_bar(
            len(self),
            title='Generating scores',
            disable=not self.verbose
        ) as bar:
            for (n_name, d_name), (network, dataset) in self.datasets.items():
                self.model.parameter = network
                for inf_name, inf in self.inferences:
                    self.model.inference = inf
                    for i in range(self.n_scores):
                        key = (n_name, inf_name, d_name, f'r{i+1}')
                        key_str = '-'.join(key[:-1])
                        if self.n_scores > 1:
                            key_str += f'-{key[-1]}'
                        
                        bar.text(f'Score {key_str}')
                        start = perf_counter()
                        result = self.model.fit(dataset)
                        runtime = perf_counter() - start
                        bar()
                        yield network, inf, dataset, result, runtime
        
        self.datasets.verbose = self.verbose
        self.inferences.verbose = self.verbose
    
    def _save(self, path: Path) -> None:
        parent_path = path.parent

        self.datasets.save(parent_path)
        self.datasets.verbose = False
        self.inferences.save(parent_path)
        self.inferences.verbose = False

        if self.path is None:
            self.datasets.path = parent_path
            self.inferences.path = parent_path
        
        for (n, i, d, r), (*_, result, runtime) in self.items():
            output = path.joinpath(n, i, d, r)
            output.mkdir(parents=True, exist_ok=True)
            result.save(output / 'result', True)
            np.save(output / 'runtime.npy', np.array([runtime]))

        self.datasets.verbose = self.verbose
        self.datasets.path = self.path
        self.inferences.verbose = self.verbose
        self.inferences.path = self.path

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
    # gen = Benchmark(verbose=True)
    # gen.save('test_benchmark')
    
    gen = Benchmark(
        path='test_benchmark',
        exclude=['Trees*/*/*/*'], 
        verbose=True
    )
    print(gen.save('test_benchmark2'))
    print(gen.save_reports('test_reports', True))
    
