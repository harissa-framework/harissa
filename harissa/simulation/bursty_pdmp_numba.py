"""
Perform simulations using the PDMP model - Fast version using Numba
NB: This module takes time to compile (~8s) but is much more efficient,
which is typically suited for large numbers of genes and/or cells
"""
import numpy as np
from harissa.simulation.simulation import Simulation
from .bursty_pdmp import BurstyPDMP
from ..utils.math import kon, kon_bound, flow
from numba import njit

kon = njit()(kon)
kon_bound = njit()(kon_bound)
flow = njit(flow)

@njit
def step(state: np.ndarray,
         basal: np.ndarray,
         inter: np.ndarray,
         d0: np.ndarray, 
         d1: np.ndarray,
         s1: np.ndarray,
         k0: np.ndarray,
         k1: np.ndarray,
         b: np.ndarray,
         tau: float | None) -> tuple[float, bool, np.ndarray]:
    """
    Compute the next jump and the next step of the
    thinning method, in the case of the bursty model.
    """
    if tau is None:
        # Adaptive thinning parameter
        tau = np.sum(kon_bound(state, basal, inter, d0, d1, s1, k0, k1))
    jump = False # Test if the jump is a true or phantom jump
    
    # 0. Draw the waiting time before the next jump
    U = np.random.exponential(scale=1/tau)
    
    # 1. Update the continuous states
    state = flow(U, state, d0, d1, s1)
    
    # 2. Compute the next jump
    v = kon(state[1], basal, inter, k0, k1) / tau # i = 1, ..., G-1 : burst of mRNA i
    v[0] = 1 - np.sum(v[1:]) # i = 0 : no change (phantom jump)
    i = np.nonzero(np.random.multinomial(1, v))[0][0]
    if i > 0:
        state[0, i] += np.random.exponential(1/b[i])
        jump = True
    
    return U, jump, state

@njit
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
               tau: float | None) -> tuple[np.ndarray, int, int]:
    """
    Exact simulation of the network in the bursty PDMP case.
    """
    states = np.empty((time_points.size, *state.shape))
    phantom_jump_count, true_jump_count = 0, 0
    t = 0.0
    # Core loop for simulation
    for i, time_point in enumerate(time_points):
        # Recording
        states[i] = flow(time_point - t, state, d0, d1, s1)

        while t < time_point:
            U, jump, state = step(state, basal, inter, d0, d1, 
                                  s1, k0, k1, b, tau)
            t += U
            if jump:
                true_jump_count += 1
            else: 
                phantom_jump_count += 1

    return states, phantom_jump_count, true_jump_count



class BurstyPDMP_Numba(BurstyPDMP):
    """
    Bursty PDMP version of the network model (promoters not described)
    """
    def __init__(self, 
                 M0: np.ndarray | None = None, P0: np.ndarray | None = None, 
                 burnin: float | None = None, 
                 thin_adapt: bool = True, verbose: bool = False) -> None:
        super().__init__(M0, P0, burnin, thin_adapt, verbose)

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
        Perform simulation of the network model (bursty PDMP version).
        """
        state, k0, k1, s1, tau = self._prepare_run(burst_frequency_min,
                                                   burst_frequency_max,
                                                   burst_size,
                                                   degradation_rna,
                                                   degradation_protein)
        if self.burn_in is not None: 
            res = simulation(state=state,
                             time_points=np.array([self.burn_in]),
                             basal=basal,
                             inter=interaction,
                             d0=degradation_rna,
                             d1=degradation_protein,
                             s1=s1, k0=k0, k1=k1, b=burst_size,
                             tau=tau)
            state = res[0][-1]
            self._display_jump_info(res[1], res[2])
        
        # Activate the stimulus
        state[1, 0] = 1
        # Final simulation with stimulus
        res = simulation(state=state,
                         time_points=time_points,
                         basal=basal,
                         inter=interaction,
                         d0=degradation_rna,
                         d1=degradation_protein,
                         s1=s1, k0=k0, k1=k1, b=burst_size,
                         tau=tau)
        states = res[0][..., 1:]
        self._display_jump_info(res[1], res[2])

        return BurstyPDMP.Result(time_points, states[:, 0], states[:, 1])