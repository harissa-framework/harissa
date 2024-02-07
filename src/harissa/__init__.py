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
from importlib.metadata import version as _version
from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from harissa.core import NetworkModel, NetworkParameter

__all__ = ['NetworkModel', 'NetworkParameter']

try:
    __version__ = _version('harissa')
except _PackageNotFoundError:
    __version__ = 'unknown version'

# Handle exceptions with user-friendly traceback:
# this may be moved later to specific end-user scripts
# ====================================================
# import sys, traceback
# def _excepthook(exc_type, exc_value, exc_traceback):
#     """Show minimal traceback for exceptions."""
#     traceback.print_exception(exc_value, limit=1)
# sys.excepthook = _excepthook
