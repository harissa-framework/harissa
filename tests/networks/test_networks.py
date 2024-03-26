import sys
from inspect import getmembers, isfunction

from harissa import NetworkParameter
import harissa.networks

def _create_test_fn(fn):
    def test():
        n_genes = 3
        network_param = fn(n_genes)

        assert isinstance(network_param, NetworkParameter)
        assert network_param.n_genes == n_genes

    return (f'{test.__name__}_{fn.__name__}', test)

for members_fn in getmembers(sys.modules['harissa.networks'], isfunction):
    name, fn = _create_test_fn(members_fn[1])
    globals()[name] = fn