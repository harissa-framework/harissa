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
from harissa.model.network_model import NetworkModel
from harissa.model.cascade import Cascade
from harissa.model.tree import Tree

__all__ = ['NetworkModel', 'Cascade', 'Tree']
