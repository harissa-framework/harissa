"""
harissa.simulation
------------------

Simulation of the network model.
"""

from harissa.simulations.bursty_pdmp.bursty_pdmp import BurstyPDMP
from harissa.simulations.approx_ode.approx_ode import ApproxODE

__all__ = ['BurstyPDMP', 'ApproxODE']