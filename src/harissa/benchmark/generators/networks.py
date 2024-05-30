from typing import (
    Dict, 
    List,
    Callable, 
    Union, 
    Optional
)

from pathlib import Path
from harissa.core import NetworkParameter
from harissa.benchmark.generators.generic import GenericGenerator
from alive_progress import alive_bar
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

class NetworksGenerator(GenericGenerator[NetworkParameter]):

    _networks : Dict[str, Callable[[], NetworkParameter]] = {}

    def __init__(self,
        include: Optional[List[str]] = None, 
        exclude: Optional[List[str]] = None,
        path: Optional[Union[str, Path]] = None
    ) -> None:
        super().__init__('networks', include, exclude, path)

    @classmethod
    def register(cls, 
        name: str, 
        network: Union[NetworkParameter, Callable[[], NetworkParameter]]
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
    
    # Alias
    @property
    def networks(self) -> Dict[str, NetworkParameter]:
        return self.items

    @classmethod
    def available_networks(cls) -> List[str]:
        return list(cls._networks.keys())

    def _load(self, path: Path) -> None:
        self._items = {}
        
        paths = self.match_rec(path)

        with alive_bar(len(paths), title='Loading Networks parameters') as bar:
            for p in paths:
                bar.text(f'Loading {p.absolute()}')
                name = str(p.relative_to(path).with_suffix(''))
                self._items[name] = NetworkParameter.load(p)
                bar()
        
    def _generate(self) -> None:
        self._items = {}
        networks = {
            k:n for k,n  in self._networks.items() 
            if self.match(k) 
        }
        with alive_bar(len(networks), title='Generating networks') as bar:
            for name, network in networks.items():
                if isinstance(network, Callable):
                    network = network()
                if isinstance(network, NetworkParameter):
                    self._items[name] = network
                else:
                    raise RuntimeError((f'{network} is not a callable'
                                        ' that returns a NetworkParameter.'))
                bar()
        
    def _save(self, path: Path) -> None:
        with alive_bar(len(self.networks)) as bar:
            for name, network in self.networks.items():
                if network.layout is None:
                    network.layout = build_pos(network.interaction)
                network.save(path / name)
            bar()

NetworksGenerator.register_defaults()

if __name__ == '__main__':
    print(NetworksGenerator.available_networks())