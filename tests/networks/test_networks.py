import sys
from inspect import getmembers, isfunction

from harissa import NetworkParameter
import harissa.networks

def _create_test_fn(fn):
    def test():
        if fn.__name__ in ['cascade', 'random_tree']:
            n_genes = 3
            network_param = fn(n_genes)
            assert network_param.n_genes == n_genes
        else:
            network_param = fn()

        assert isinstance(network_param, NetworkParameter)

    return (f'{test.__name__}_{fn.__name__}', test)

for members_fn in getmembers(sys.modules['harissa.networks'], isfunction):
    name, fn = _create_test_fn(members_fn[1])
    globals()[name] = fn