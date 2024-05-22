"""
harissa.inference
-----------------

Inference of the network model.
"""
from harissa.inference.hartree import Hartree
from harissa.inference.cardamom import Cardamom

# Default inference method
default_inference = Hartree

__all__ = ['Hartree', 'Cardamom']
