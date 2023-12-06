"""
Harissa
=======

Tools for mechanistic gene network inference from single-cell data
------------------------------------------------------------------

Mechanistic model-based gene network inference using
a self-consistent proteomic field (SCPF) approximation.
It is analogous to the unrestricted Hartree approximation
in quantum mechanics, applied to gene expression modeled
as a piecewise-deterministic Markov process (PDMP).

The package also includes a simulation module to generate
single-cell data with transcriptional bursting.

Author: Ulysse Herbach (ulysse.herbach@inria.fr)
"""
from harissa.model import NetworkModel
from harissa.parameter import NetworkParameter
from harissa.utils.networks import cascade, random_tree

__all__ = ['NetworkModel', 'NetworkParameter', 'cascade', 'random_tree']
