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
from harissa.core import NetworkModel, NetworkParameter, Dataset

__all__ = ['NetworkModel', 'NetworkParameter', 'Dataset']

try:
    __version__ = _version('harissa')
except Exception:
    __version__ = 'unknown version'

try:
    from alive_progress.animations.spinners import (
        bouncing_spinner_factory as _bouncing_spinner_factory
    )
    from alive_progress import config_handler as _config_handler

    _config_handler.set_global(
        bar='smooth',
        spinner=_bouncing_spinner_factory('🌶', 6, hide=False),
        unknown='horizontal',
        # dual_line=True,
        receipt=False,
        force_tty=True,
        length=20,
        max_cols=100
    )
except ImportError:
    pass

# Handle exceptions with user-friendly traceback:
# this may be moved later to specific end-user scripts
# ====================================================
# import sys, traceback
# def _excepthook(exc_type, exc_value, exc_traceback):
#     """Show minimal traceback for exceptions."""
#     traceback.print_exception(exc_value, limit=1)
# sys.excepthook = _excepthook
