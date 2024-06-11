from typing import (
    Dict, 
    List,
    Tuple,
    Callable, 
    Union, 
    Optional,
    TypeAlias
)

from collections.abc import Iterator

from pathlib import Path
from harissa.core import NetworkParameter
from harissa.benchmark.generators.generic import GenericGenerator
from harissa.utils.progress_bar import alive_bar
from functools import wraps

from harissa.plot import build_pos
import harissa.networks as networks

def normalize(func):
    wraps(func)
    def wrapper(*args, **kwargs):
        net = func(*args, **kwargs)
        net.creation_rna[:] = net.degradation_rna * net.rna_scale()
        net.creation_protein[:] = net.degradation_protein * net.protein_scale()

        return net
    return wrapper


@normalize
def bn8():
    net = networks.bn8()
    net.degradation_rna[:] = 0.25
    net.degradation_protein[:] = 0.05
    
    return net

@normalize
def cn5():
    net = networks.cn5()
    net.degradation_rna[:] = 0.5
    net.degradation_protein[:] = 0.1
    
    return net

@normalize
def fn4():
    net = networks.fn4()
    net.degradation_rna[:] = 1
    net.degradation_protein[:] = 0.2
    net.d[:] /= 5

    return net

@normalize
def fn8():
    net = networks.fn8()
    net.degradation_rna[:] = 0.4
    net.degradation_protein[:] = 0.08

    return net

@normalize
def tree(n_genes):
    net = networks.random_tree(n_genes)

    net.degradation_rna[:] = 1
    net.degradation_protein[:] = 0.2
    net.d[:] /= 4

    return net

K: TypeAlias = str
V: TypeAlias = NetworkParameter
class NetworksGenerator(GenericGenerator[K, V]):

    _networks : Dict[str, Union[V, Callable[[], V]]] = {}

    def __init__(self,
        include: Optional[List[str]] = None, 
        exclude: Optional[List[str]] = None,
        path: Optional[Union[str, Path]] = None,
        verbose: bool = False
    ) -> None:
        super().__init__('networks', include, exclude, path, verbose)

    @classmethod
    def register(cls, 
        name: str, 
        network: Union[V, Callable[[], V]]
    ) -> None:
        if isinstance(network, (NetworkParameter, Callable)):
            if name not in cls._networks:
                cls._networks[name] = network
            else:
                raise ValueError((f'{name} is already taken. '
                                  f'Cannot register {network}.'))
        else:
            raise TypeError(('network must be a NetworkParameter or a '
                             'callable that returns a NetworkParameter.'))
    
    @classmethod
    def register_defaults(cls) -> None:
        cls.register('BN8', bn8)
        cls.register('CN5', cn5)
        cls.register('FN4', fn4)
        cls.register('FN8', fn8)
        for g in [5, 10, 20, 50, 100]:
            cls.register(f'Trees{g}', lambda: tree(g))

    @classmethod
    def unregister_all(cls) -> None:
        cls._networks = {}

    @classmethod
    def available_networks(cls) -> List[str]:
        return list(cls._networks.keys())

    def _load_keys(self, path: Path) -> Iterator[K]:
        for p in self.match_rec(path):
            key = str(
                p
                .relative_to(path / self.sub_directory_name)
                .with_suffix('')
            )
            yield key

    def _load(self, path: Path) -> Iterator[Tuple[K, V]]:
        paths = self.match_rec(path)
        with alive_bar(
            len(paths), 
            title='Loading Networks parameters',
            disable=not self.verbose
        ) as bar:
            for p in paths:
                bar.text(f'Loading {p.absolute()}')
                name = str(
                    p
                    .relative_to(path / self.sub_directory_name)
                    .with_suffix('')
                )
                network = NetworkParameter.load(p)
                bar()
                yield name, network

    def _load_values(self, path: Path) -> Iterator[V]:
        paths = self.match_rec(path)
        with alive_bar(
            len(paths), 
            title='Loading Networks parameters',
            disable=not self.verbose
        ) as bar:
            for p in paths:
                bar.text(f'Loading {p.absolute()}')
                network = NetworkParameter.load(p)
                bar()
                yield network
        

    def _generate_keys(self) -> Iterator[K]:
        for key in self._networks.keys():
            if self.match(key):
                yield key
        
    def _generate(self) -> Iterator[Tuple[K, V]]:
        networks = {
            k:n for k,n  in self._networks.items() 
            if self.match(k) 
        }
        with alive_bar(
            len(networks), 
            title='Generating networks',
            disable=not self.verbose
        ) as bar:
            for name, network in networks.items():
                bar.text(name)
                if isinstance(network, Callable):
                    network = network()
                if not isinstance(network, NetworkParameter):
                    raise RuntimeError((f'{network} is not a callable'
                                        ' that returns a NetworkParameter.'))
                bar()
                yield name, network
    
    def _generate_values(self) -> Iterator[V]:
        networks = {
            k:n for k,n in self._networks.items() 
            if self.match(k) 
        }
        with alive_bar(
            len(networks), 
            title='Generating networks',
            disable=not self.verbose
        ) as bar:
            for name, network in networks.items():
                bar.text(name)
                if isinstance(network, Callable):
                    network = network()
                if not isinstance(network, NetworkParameter):
                    raise RuntimeError((f'{network} is not a callable'
                                        ' that returns a NetworkParameter.'))
                bar()
                yield network
        
    def _save(self, path: Path) -> None:
        for name, network in self.items():
            if network.layout is None:
                network.layout = build_pos(network.interaction)
            network.save(path / name)

NetworksGenerator.register_defaults()

if __name__ == '__main__':
    print(NetworksGenerator.available_networks())
    for name, network in NetworksGenerator():
        print(name, network)