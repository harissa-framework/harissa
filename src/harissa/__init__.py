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

# Import utils as harissa modules
import harissa.utils.graphics as graphics
import harissa.utils.networks as networks
import harissa.utils.processing as processing
import harissa.utils.stat as stat

__all__ = ['NetworkModel', 'NetworkParameter', 'cascade', 'random_tree',
    'graphics', 'networks', 'processing', 'stat']
