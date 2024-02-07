"""
harissa.inference.hartree
-------------------------

Mechanistic model-based gene network inference using
a self-consistent proteomic field (SCPF) approximation.
It is analogous to the unrestricted Hartree approximation
in quantum mechanics, applied to gene expression modeled
as a piecewise-deterministic Markov process (PDMP).
"""
from harissa.inference.hartree.base import Hartree

__all__ = ["Hartree"]
