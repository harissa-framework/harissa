from typing import (
    List,
    Tuple,
    Union,
    Optional
)

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
from harissa.benchmark.generators.generic import Iterator

from harissa.plot.plot_benchmark import plot_benchmark
from harissa.utils.progress_bar import alive_bar

K = Tuple[str, str, str, str]
V = Tuple[
    NetworkParameter,
    Tuple[Inference, npt.NDArray[np.float64]],
    Dataset,
    Inference.Result,
    float
]
class Benchmark(GenericGenerator[K, V]):
    """
    Benchmark, the generator of scores.
    """
    def __init__(self,
        n_run: Union[int, List[str]] = 1, 
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
        """
        Set the path to self and to its generators
        """
        super()._set_path(path)
        for generator in self._generators:
            generator.path = path

    # Aliases    
    @property
    def datasets(self):
        """
        Datasets generator
        """
        return self._generators[0]
    
    @property
    def inferences(self):
        """
        Inference methods generator
        """
        return self._generators[1]
    
    @property
    def networks(self):
        """
        Networks generator
        """
        return self.datasets.networks
    
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
        network, dataset = self.datasets[key[0], key[2]]
        inf = self.inferences[key[1]]

        path = self._to_path(key)
        
        if not path.exists():
            raise KeyError(f'{key} is invalid. {path} does not exist.')
        
        result = inf[0].Result.load(path / 'result.npz', load_extra=True)
        runtime_in_sec = np.load(path / 'runtime.npy').item()

        return network, inf, dataset, result, runtime_in_sec

    def _load_keys(self) -> Iterator[K]:
        """
        Load all the keys

        Yields
        ------
        K
        """
        for dataset_key in self.datasets.keys():
            for inf_name in self.inferences.keys():
                run_key = (dataset_key[0], inf_name, dataset_key[1])
                run_dir = self._to_path(run_key)
                for run_path in run_dir.iterdir():
                    key = (*run_key, run_path.stem)
                    if self.match(key):
                        yield key

    def _get_n_run(self) -> List[str]:
        """
        Get the run names.
        """
        if isinstance(self.n_run, int):
            return [f'r{i+1}' for i in range(self.n_run)]
        else:
            return self.n_run

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
        n_run = self._get_n_run()
        if key[-1] not in n_run:
            raise KeyError(
                f'{key} is invalid. {key[-1]} must be inside {n_run}.'
            )

        network, dataset = self.datasets[key[0], key[2]]
        inf = self.inferences[key[1]]
        
        self._model.parameter = network
        self._model.inference = inf[0]
        
        start = perf_counter()
        result = self._model.fit(dataset)
        runtime_in_sec = perf_counter() - start
            
        return network, inf, dataset, result, runtime_in_sec

    def _generate_keys(self) -> Iterator[K]:
        """
        Generate all the keys

        Yields
        ------
        K
        """
        for dataset_key in self.datasets.keys():
            for inf_name in self.inferences.keys():
                for run_name in self._get_n_run():
                    key = (dataset_key[0], inf_name, dataset_key[1], run_name)
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
        key, (network, inf, dataset, result, runtime) = item

        output = self._to_path(key, path)
        output.mkdir(parents=True, exist_ok=True)

        result.save(output / 'result', True)
        np.save(output / 'runtime.npy', np.array(runtime))

        keys = [(key[0], key[2]), key[1]]
        values = [(network, dataset), inf]

        for generator, key, value in zip(self._generators, keys, values):
            generator.save_item(path, key, value)


    def reports(self, show_networks: bool = False) -> List:
        """
        Generate reports (figures) of the benchmark.

        Parameters
        ----------
        show_networks
            Show networks in the figures, by default False

        Returns
        -------
            3 matplotlib figures
        """
        networks, inferences = set(), set()
        for network, inference, *_ in self.keys():
            networks.add(network)
            inferences.add(inference)

        return plot_benchmark(
            self, 
            sorted(networks), 
            sorted(inferences), 
            show_networks
        )

    def save_reports(self,
        path: Union[str, Path],
        archive_format: Optional[str] = None,
        show_networks: bool = False,
        save_all: bool = False
    ) -> Path:
        """
        Save the generated reports

        Parameters
        ----------
        path
            path where to save
        archive_format
            The format to archive the path, by default None.
            If None the path is not archived.
            The available formats are `zip`, `tar`, `gztar`, `bztar` 
            or `xztar`. 
            The archive format is pass to the function `make_archive`, see
            more informations at https://docs.python.org/3/library/shutil.html#shutil.make_archive
        show_networks
            Show networks in the figures, by default False
        save_all
            Save also scores, datasets, networks and inference methods 
            of the benchmark, by default False.

        Returns
        -------
            The absolute path containing the saved items
        """
        path = Path(path).with_suffix('')
        generators = [self, *self._generators, self.networks]
        old_paths = [gen.path for gen in generators]

        if archive_format is not None:
            with TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)
                if save_all:
                    self.save(tmp_path)
                    self.path = tmp_path
                    tmp_path = tmp_path / 'reports'

                tmp_path.mkdir(parents=True, exist_ok=True)
                for fig in self.reports(show_networks):
                    fpath = (tmp_path / fig.get_suptitle()).with_suffix('.pdf')
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
                path = path / 'reports'
            
            path.mkdir(parents=True, exist_ok=True)
            for fig in self.reports(show_networks):
                fpath = (path / fig.get_suptitle()).with_suffix('.pdf')
                fig.savefig(fpath)
            
            if save_all:
                path = path.parent

        for generator, old_path in zip(generators, old_paths):
            generator.path = old_path
        
        return path.absolute()
    

if __name__ == '__main__':
    benchmark = Benchmark()
    # benchmark.datasets.path = 'test_datasets2'
    # benchmark.networks.include = ['BN8']
    print(benchmark.save_reports('test_benchmark', None, True, True))
    
    benchmark = Benchmark(path='test_benchmark.zip')
    benchmark.networks.exclude = ['Trees*']
    print(benchmark.save('test_benchmark2'))
    print(benchmark.save_reports('test_reports', show_networks=True))
    
