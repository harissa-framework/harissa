"""
Perform simulations using the ODE model
"""
from .simulation import Simulation
from ..utils.math import kon
import numpy as np

class ApproxODE(Simulation):
    """
    ODE version of the network model (very rough approximation of the PDMP)
    """
    def __init__(self, M0=None, P0=None, burnin=None, verbose=False):
        self.M0 : np.ndarray | None = M0
        self.P0 : np.ndarray | None = P0
        self.burn_in : float | None = burnin
        self.state : np.ndarray | None = None
        self.is_verbose : bool  = verbose

    def run(self, 
            time_points: np.ndarray, 
            burst_frequency_min: np.ndarray, 
            burst_frequency_max: np.ndarray, 
            burst_size: np.ndarray, 
            degradation_rna: np.ndarray, 
            degradation_protein: np.ndarray,
            basal: np.ndarray, 
            interaction: np.ndarray) -> Simulation.Result:
        """
        Perform simulation of the network model (ODE version).
        This is the slow-fast limit of the PDMP model, which is only
        relevant when promoters & mRNA are much faster than proteins.
        p: solution of a nonlinear ODE system involving proteins only
        m: mean mRNA levels given protein levels (quasi-steady state)
        """
        nb_genes = basal.size
        
        k0 = burst_frequency_min * degradation_rna
        k1 = burst_frequency_max * degradation_rna

        # Normalize protein scales
        s1 = degradation_protein * burst_size / burst_frequency_max

        # Simulation parameter
        euler_step = 1e-3/np.max(degradation_protein)

        # Default state
        types = [('M','float'), ('P','float')]
        self.state = np.array([(0,0) for i in range(nb_genes)], dtype=types)
        # state  = np.zeros((2, nb_genes))

        # Burnin simulation without stimulus
        if self.M0 is not None:
            self.state['M'][1:] = self.M0[1:]
        if self.P0 is not None: 
            self.state['P'][1:] = self.P0[1:]
        if self.burn_in is not None: 
            self._simulation(time_points=np.array([self.burn_in]),
                             basal=basal,
                             inter=interaction,
                             d0=degradation_rna,
                             d1=degradation_protein,
                             s1=s1,
                             k0=k0,
                             k1=k1,
                             b=burst_size,
                             dt=euler_step)
        
        # Activate the stimulus
        self.state['P'][0] = 1

        # Final simulation with stimulus
        res = self._simulation(time_points=time_points,
                               basal=basal,
                               inter=interaction,
                               d0=degradation_rna,
                               d1=degradation_protein,
                               s1=s1,
                               k0=k0,
                               k1=k1,
                               b=burst_size,
                               dt=euler_step)
        
        return Simulation.Result(time_points, res['M'], res['P'])
        # return Simulation.Result(time_points, res[:, 0], res[:, 1])

    def _simulation(self, time_points, basal, inter, 
                   d0, d1, s1, k0, k1, b, dt):
        """
        Simulation of the deterministic limit model, which is relevant when
        promoters and mRNA are much faster than proteins.
        1. Nonlinear ODE system involving proteins only
        2. Mean level of mRNA given protein levels
        """
        nb_genes = basal.size
        if np.size(time_points) > 1:
            dt = np.min([dt, np.min(time_points[1:] - time_points[:-1])])
        types = [('M','float'), ('P','float')]
        sim = []
        T, c = 0, 0
        # Core loop for simulation and recording
        for t in time_points:
            while T < t:
                self._step(basal, inter, d0, d1, s1, k0, k1, b, dt)
                T += dt
                c += 1
            M, P = self.state['M'], self.state['P']
            sim += [np.array([(M[i],P[i]) for i in range(1, nb_genes)], 
                             dtype=types)]
        
        # Display info about steps
        if self.is_verbose:
            if c > 0:
                print(f'ODE simulation used {c} steps (step size = {dt:.5f})')
            else: 
                print('ODE simulation used no step')
        return np.array(sim)

    def _step(self, basal, inter, d0, d1, s1, k0, k1, b, dt):
        """
        Euler step for the deterministic limit model.
        """
        m, p = self.state['M'], self.state['P']
        a = kon(p, basal, inter, k0, k1) / d0 # a = kon/d0, b = koff/s0
        m_new = a/b # Mean level of mRNA given protein levels
        p_new = (1 - dt*d1)*p + dt*s1*m_new # Protein-only ODE system
        m_new[0], p_new[0] = m[0], p[0] # Discard stimulus
        
        self.state['M'], self.state['P'] = m_new, p_new