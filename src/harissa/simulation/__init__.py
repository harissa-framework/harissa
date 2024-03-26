"""
harissa.simulation
------------------

Simulation of the network model.
"""
from harissa.simulation.bursty_pdmp import BurstyPDMP
from harissa.simulation.approx_ode import ApproxODE

# Default simulation method
default_simulation = BurstyPDMP

__all__ = ['BurstyPDMP', 'ApproxODE']
