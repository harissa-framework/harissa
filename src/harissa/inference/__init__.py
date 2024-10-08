"""
Inference of the network model.
"""
from harissa.inference.hartree import Hartree
from harissa.inference.cardamom import Cardamom
from harissa.inference.pearson import Pearson

# Default inference method
default_inference = Cardamom

__all__ = ['Hartree', 'Cardamom', 'Pearson']
