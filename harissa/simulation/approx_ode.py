"""
Perform simulations using the ODE model
"""
from .bursty_pdmp import Simulation
from ..utils.math import kon 
import numpy as np

def step(state: np.ndarray,
         basal: np.ndarray,
         inter: np.ndarray,
         d0: np.ndarray, d1: np.ndarray,
         s1: np.ndarray, k0: np.ndarray, k1: np.ndarray, b: np.ndarray,
         dt: float) -> np.ndarray:
    """
    Euler step for the deterministic limit model.
    """
    m, p = state
    a = kon(p, basal, inter, k0, k1) / d0 # a = kon/d0, b = koff/s0
    m_new = a/b # Mean level of mRNA given protein levels
    p_new = (1 - dt*d1)*p + dt*s1*m_new # Protein-only ODE system
    m_new[0], p_new[0] = m[0], p[0] # Discard stimulus
    
    return np.vstack((m_new, p_new))


def simulation(state: np.ndarray,
               time_points: np.ndarray,
               basal: np.ndarray,
               inter: np.ndarray,
               d0: np.ndarray,
               d1: np.ndarray,
               s1: np.ndarray,
               k0: np.ndarray,
               k1: np.ndarray,
               b: np.ndarray,
               dt:float) -> np.ndarray:
    """
    Simulation of the deterministic limit model, which is relevant when
    promoters and mRNA are much faster than proteins.
    1. Nonlinear ODE system involving proteins only
    2. Mean level of mRNA given protein levels
    """
    states = np.empty((time_points.size, *state.shape))
    if time_points.size > 1:
        dt = np.min([dt, np.min(time_points[1:] - time_points[:-1])])
    t, step_count = 0.0, 0
    # Core loop for simulation and recording
    for i, time_point in enumerate(time_points):
        while t < time_point:
            state = step(state, basal, inter, d0, d1, s1, k0, k1, b, dt)
            t += dt
            step_count += 1
        states[i] = state
    
    return states, step_count, dt

class ApproxODE(Simulation):
    """
    ODE version of the network model (very rough approximation of the PDMP)
    """
    def __init__(self, M0=None, P0=None, burnin=None, verbose=False):
        self.M0 : np.ndarray | None = M0
        self.P0 : np.ndarray | None = P0
        self.burn_in : float | None = burnin
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
        state, k0, k1, s1, euler_step = self._prepare_run(burst_frequency_min, 
                                                          burst_frequency_max, 
                                                          burst_size, 
                                                          degradation_rna, 
                                                          degradation_protein)

        # Burnin simulation without stimulus
        if self.burn_in is not None: 
            res = simulation(state=state,
                               time_points=np.array([self.burn_in]),
                               basal=basal,
                               inter=interaction,
                               d0=degradation_rna,
                               d1=degradation_protein,
                               s1=s1, k0=k0, k1=k1,
                               b=burst_size,
                               dt=euler_step)
            state = res[0][-1]
            self._display_step_info(res[1], res[2])
        
        # Activate the stimulus
        state[1, 0] = 1

        # Final simulation with stimulus
        res = simulation(state=state, 
                         time_points=time_points,
                         basal=basal,
                         inter=interaction,
                         d0=degradation_rna,
                         d1=degradation_protein,
                         s1=s1, k0=k0, k1=k1,
                         b=burst_size,
                         dt=euler_step)
        states = res[0][..., 1:]
        self._display_step_info(res[1], res[2])
        
        return Simulation.Result(time_points, states[:, 0], states[:, 1])
    
    def _prepare_run(self, 
                     burst_frequency_min, burst_frequency_max, burst_size, 
                     degradation_rna, degradation_protein):
        
        k0 = burst_frequency_min * degradation_rna
        k1 = burst_frequency_max * degradation_rna

        # Normalize protein scales
        s1 = degradation_protein * burst_size / burst_frequency_max

        # Simulation parameter
        euler_step = 1e-3/np.max(degradation_protein)

        # Default state: row 0 <-> M, row 1 <-> P
        nb_genes = degradation_rna.size
        state = np.zeros((2, nb_genes))

        if self.M0 is not None:
            state[0, 1:] = self.M0[1:]
        if self.P0 is not None: 
            state[1, 1:] = self.P0[1:]

        return state, k0, k1, s1, euler_step
    
    def _display_step_info(self, step_count: int, dt: float):
        """
        Display info about steps
        """
        if self.is_verbose:
            if step_count > 0:
                print(f'ODE simulation used {step_count} steps ' 
                      f'(step size = {dt:.5f})')
            else: 
                print('ODE simulation used no step')