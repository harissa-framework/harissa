"""
Perform simulations using the PDMP model
"""
from .simulation import Simulation
from ..utils.math import kon, kon_bound, flow

import numpy as np
# from scipy.special import expit

# def kon(p: np.ndarray, 
#         basal: np.ndarray, 
#         inter: np.ndarray, 
#         k0: np.ndarray, 
#         k1: np.ndarray) -> float:
#     """
#     Interaction function kon (off->on rate), given protein levels p.
#     """
#     sigma = expit(basal + p @ inter)
#     k_on = (1 - sigma) * k0 + sigma * k1
#     k_on[0] = 0 # Ignore stimulus
#     return k_on

# def kon_bound(state: np.ndarray, 
#               basal: np.ndarray, 
#               inter: np.ndarray, 
#               d0: np.ndarray, 
#               d1: np.ndarray, 
#               s1: np.ndarray, 
#               k0: np.ndarray, 
#               k1: np.ndarray) -> float:
#     """
#     Compute the current kon upper bound.
#     """
#     m, p = state
#     # Explicit upper bound for p
#     time = np.log(d0/d1)/(d0-d1) # vector of critical times
#     p_max = p + (s1/(d0-d1))*m*(np.exp(-time*d1) - np.exp(-time*d0))
#     p_max[0] = p[0] # Discard stimulus
#     # Explicit upper bound for Kon
#     sigma = expit(basal + p_max @ ((inter > 0) * inter))
#     k_on = (1-sigma)*k0 + sigma*k1 + 1e-10 # Fix precision errors
#     k_on[0] = 0 # Ignore stimulus
#     return k_on

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
    G = basal.size # Genes plus stimulus
    v = kon(state[1], basal, inter, k0, k1) / tau # i = 1, ..., G-1 : burst of mRNA i
    v[0] = 1 - np.sum(v[1:]) # i = 0 : no change (phantom jump)
    i = np.random.choice(G, p=v)
    if i > 0:
        state[0, i] += np.random.exponential(1/b[i])
        jump = True
    
    return U, jump, state

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
    t, t_old, state_old = 0.0, 0.0, state
    # Core loop for simulation
    for i, time_point in enumerate(time_points):
        while t < time_point:
            t_old, state_old = t, state
            U, jump, state = step(state, basal, inter, d0, d1, 
                                  s1, k0, k1, b, tau)
            t += U
            if jump:
                true_jump_count += 1
            else: 
                phantom_jump_count += 1
        # Recording
        states[i] = flow(time_point - t_old, state_old, d0, d1, s1)


    return states, phantom_jump_count, true_jump_count

class BurstyPDMP(Simulation):
    """
    Bursty PDMP version of the network model (promoters not described)
    """

    def __init__(self, 
                 M0: np.ndarray | None = None, P0: np.ndarray | None = None, 
                 burnin: float | None = None, 
                 thin_adapt: bool = True, verbose: bool = False) -> None:
        self.M0 : np.ndarray | None = M0
        self.P0 : np.ndarray | None = P0
        self.burn_in : float | None = burnin
        self.thin_adapt : bool  = thin_adapt
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
        Perform simulation of the network model (bursty PDMP version).
        """
        state, k0, k1, s1, tau = self._prepare_run(burst_frequency_min,
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
                             s1=s1, k0=k0, k1=k1, b=burst_size,
                             tau=tau)
            state = res[0][-1] # Update the current state
            self._display_jump_info(res[1], res[2])
        
        # Activate the stimulus
        state[1, 0] = 1.0
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
        
        return Simulation.Result(time_points, states[:, 0], states[:, 1])
    
    def _prepare_run(self, 
                     burst_frequency_min, burst_frequency_max, burst_size, 
                     degradation_rna, degradation_protein):
        k0 = burst_frequency_min * degradation_rna
        k1 = burst_frequency_max * degradation_rna

        # Normalize protein scales
        s1 = degradation_protein * burst_size / burst_frequency_max

        # Thinning parameter
        tau = None if self.thin_adapt else np.sum(k1[1:])

        nb_genes = degradation_rna.size

        # Default state: row 0 <-> M, row 1 <-> P
        state = np.zeros((2, nb_genes))

        if self.M0 is not None:
            state[0, 1:] = self.M0[1:]
        if self.P0 is not None: 
            state[1, 1:] = self.P0[1:]

        return state, k0, k1, s1, tau
    
    def _display_jump_info(self, phantom_jump_count, true_jump_count):
        # Display info about jumps
        if self.is_verbose:
            total_jump = phantom_jump_count + true_jump_count
            if total_jump > 0:
                print(f'Exact simulation used {total_jump} jumps ' 
                      f'including {phantom_jump_count} phantom jumps ' 
                      f'({100*phantom_jump_count/total_jump : .2f}%)')
            else: 
                print('Exact simulation used no jump')