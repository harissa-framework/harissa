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
        include: List[str] = ['*'], 
        exclude: List[str] = [],
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
    
    def _load_value(self, key: K) -> V:
        network_path = self._to_path(key).with_suffix('.npz')
        return NetworkParameter.load(network_path)
    
    def _generate_value(self, key):
        network = self._networks[key]
        if isinstance(network, Callable):
            network = network()
        if not isinstance(network, NetworkParameter):
            raise RuntimeError((f'{network} is not a callable'
                                ' that returns a NetworkParameter.'))
        return network

    def _generate_keys(self) -> Iterator[K]:
        for key in self._networks.keys():
            if self.match(key):
                yield key

    def _save_item(self, path: Path, item: Tuple[K, V]):
        key, network = item
        output = self._to_path(key, path).with_suffix('.npz')
        output.parent.mkdir(parents=True, exist_ok=True)
        
        if network.layout is None:
            network.layout = build_pos(network.interaction)
        
        network.save(output)

NetworksGenerator.register_defaults()

if __name__ == '__main__':
    an = NetworksGenerator.available_networks()
    print(an)
    gen = NetworksGenerator(path='test_benchmark', include=['BN8', 'FN4'])
    for name, network in gen.items():
        print(name, network)

    print(gen[an[0]])