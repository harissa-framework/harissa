"""
harissa.simulation
------------------

Simulation of the network model.
"""

from harissa.simulation.simulation import Simulation
from harissa.simulation.bursty_pdmp.bursty_pdmp import BurstyPDMP
from harissa.simulation.approx_ode.approx_ode import ApproxODE

__all__ = ['Simulation', 'BurstyPDMP', 'ApproxODE']