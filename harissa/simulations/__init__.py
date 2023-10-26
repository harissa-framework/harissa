"""
harissa.simulation
------------------

Simulation of the network model.
"""

from harissa.simulations.simulation import Simulation
from harissa.simulations.bursty_pdmp.bursty_pdmp import BurstyPDMP
from harissa.simulations.approx_ode.approx_ode import ApproxODE

__all__ = ['Simulation', 'BurstyPDMP', 'ApproxODE']