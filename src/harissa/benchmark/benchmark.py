from typing import (
    List,
    Tuple,
    Union, 
    Optional,
    TypeAlias
)
from collections.abc import Iterator

from pathlib import Path
from tempfile import TemporaryDirectory
from shutil import make_archive
from time import perf_counter

import numpy as np
import numpy.typing as npt

from harissa.core import NetworkModel, NetworkParameter, Inference, Dataset
from harissa.benchmark.generators import (
    GenericGenerator,
    DatasetsGenerator,
    InferencesGenerator
)

from harissa.plot.plot_benchmark import plot_benchmark
from harissa.utils.progress_bar import alive_bar

K : TypeAlias = Tuple[str, str, str, str]
V : TypeAlias = Tuple[
    NetworkParameter, 
    Tuple[Inference, npt.NDArray[np.float64]], 
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
        self._generators = [
            DatasetsGenerator(verbose=verbose),
            InferencesGenerator(verbose=verbose)
        ]

        super().__init__('scores', include, exclude, path, verbose)

        self._model = NetworkModel()
        self.n_run = n_run

    def _set_path(self, path: Path):
        super()._set_path(path)
        for generator in self._generators:
            generator.path = path
            
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
    
    def _load_value(self, key: K) -> V:
        network, dataset = self.datasets[key[0], key[2]]
        inf = self.inferences[key[1]]

        result_path = self._to_path(key)
        result = inf[0].Result.load(
            result_path / 'result.npz', 
            load_extra=True
        )
        runtime = np.load(result_path / 'runtime.npy')

        return network, inf, dataset, result, runtime

    def _load_keys(self) -> Iterator[K]:
        for dataset_key in self.datasets.keys():
            for inf_name in self.inferences.keys():
                run_key = (dataset_key[0], inf_name, dataset_key[1])
                run_dir = self._to_path(run_key)
                for run_path in run_dir.iterdir():
                    key = (*run_key, run_path.stem)
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

    def _save_item(self, path: Path, item: Tuple[K, V]):
        key, (network, inf, dataset, result, runtime) = item

        output = self._to_path(key, path)
        output.mkdir(parents=True, exist_ok=True)

        result.save(output / 'result', True)
        np.save(output / 'runtime.npy', np.array([runtime]))

        keys = [(key[0], key[2]), key[1]]
        values = [(network, dataset), inf]

        for generator, key, value in zip(self._generators, keys, values):
            generator.save_item(path, key, value)


    def reports(self, show_networks=False):
        return plot_benchmark(self, show_networks)

    def save_reports(self,
        path: Union[str, Path],
        archive_format: Optional[str] = None,
        show_networks: bool = False,
        save_all: bool = False
    ) -> Path:
        path = Path(path)
        generators = [self, *self._generators, self.networks]
        old_paths = [gen.path for gen in generators]

        fnames = ['general.pdf', 'directed.pdf', 'undirected.pdf']
        fnames = map(lambda fname: Path(fname), fnames)
        if save_all:
            fnames = map(lambda fname: Path('reports') / fname, fnames)

        if archive_format is not None:
            with TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)
                if save_all:
                    self.save(tmp_path)
                    self.path = tmp_path

                for fig, fname in zip(self.reports(show_networks), fnames):
                    fpath = tmp_path / fname
                    fpath.parent.mkdir(parents=True, exist_ok=True)
                    fig.savefig(fpath)
                
                with alive_bar(
                    title='Archiving', 
                    monitor=False, 
                    stats= False
                ) as bar:
                    path=Path(make_archive(str(path), archive_format, tmp_dir))
                    bar()
        else:
            if save_all:
                self.save(path)
                self.path = path
                
            for fig, fname in zip(self.reports(show_networks), fnames):
                fpath = path / fname
                fpath.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(fpath)

        for generator, old_path in zip(generators, old_paths):
            generator.path = old_path
        
        return path
    

if __name__ == '__main__':
    benchmark = Benchmark()
    # benchmark.datasets.path = 'test_benchmark.zip'
    # benchmark.networks.include = ['BN8']
    benchmark.save_reports('test_benchmark', 'zip', True, True)
    
    benchmark = Benchmark(path='test_benchmark.zip')
    benchmark.networks.exclude = ['Trees*']
    print(benchmark.save('test_benchmark2'))
    print(benchmark.save_reports('test_reports', show_networks=True))
    
