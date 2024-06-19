"""
harissa.networks
----------------

Various network patterns.
"""
from harissa.networks.cascade import cascade
from harissa.networks.random_tree import random_tree
from harissa.networks.bn8 import bn8
from harissa.networks.cn5 import cn5
from harissa.networks.fns import fn4, fn8

__all__ = ['cascade', 'random_tree', 'bn8', 'cn5', 'fn4', 'fn8']
