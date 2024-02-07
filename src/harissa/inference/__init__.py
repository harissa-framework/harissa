"""
harissa.inference
-----------------

Inference of the network model.
"""
from harissa.inference.hartree import Hartree

# Default inference method
default_inference = Hartree

__all__ = ["Hartree"]
