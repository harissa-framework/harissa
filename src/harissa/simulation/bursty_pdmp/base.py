"""
Perform simulations using the PDMP model
"""
from typing import Tuple, Optional
import numpy as np
from harissa.core.parameter import NetworkParameter
from harissa.core.simulation import Simulation
from harissa.simulation.bursty_pdmp.utils import kon, kon_bound, flow

def _kon_jit(p: np.ndarray, 
             basal: np.ndarray, 
             inter: np.ndarray, 
             k0: np.ndarray, 
             k1: np.ndarray) -> np.ndarray:
    """
    Interaction function kon (off->on rate), given protein levels p.
    """
    phi = np.exp(basal + p @ inter)
    k_on = (k0 + k1*phi)/(1 + phi)
    k_on[0] = 0 # Ignore stimulus
    return k_on

def _kon_bound_jit(state: np.ndarray, 
                   basal: np.ndarray, 
                   inter: np.ndarray, 
                   d0: np.ndarray, 
                   d1: np.ndarray, 
                   s1: np.ndarray, 
                   k0: np.ndarray, 
                   k1: np.ndarray) -> np.ndarray:
    """
    Compute the current kon upper bound.
    """
    m, p = state
    # Explicit upper bound for p
    time = np.log(d0/d1)/(d0-d1) # vector of critical times
    p_max = p + (s1/(d0-d1))*m*(np.exp(-time*d1) - np.exp(-time*d0))
    p_max[0] = p[0] # Discard stimulus
    # Explicit upper bound for Kon
    phi = np.exp(basal + p_max @ ((inter > 0) * inter))
    k_on = (k0 + k1*phi)/(1 + phi)
    k_on[0] = 0 # Ignore stimulus
    return k_on

_flow_jit = None

def step(state: np.ndarray,
         basal: np.ndarray,
         inter: np.ndarray,
         d0: np.ndarray, 
         d1: np.ndarray,
         s1: np.ndarray,
         k0: np.ndarray,
         k1: np.ndarray,
         b: np.ndarray,
         tau: Optional[float]) -> Tuple[float, bool, np.ndarray]:
    """
    Compute the next jump and the next step of the
    thinning method, in the case of the bursty model.
    """
    if tau is None:
        # Adaptive thinning parameter
        tau = np.sum(kon_bound(state, basal, inter, d0, d1, s1, k0, k1))
    
    # 0. Draw the waiting time before the next jump
    U = np.random.exponential(scale=1/tau)
    
    # 1. Update the continuous states
    state = flow(U, state, d0, d1, s1)
    
    # 2. Compute the next jump    
    # Deal robustly with precision errors
    v = kon(state[1], basal, inter, k0, k1) # i = 1, ..., G-1 : burst of mRNA i
    v[1:] /= tau # i = 1, ..., G-1 : burst of mRNA i
    v[0] = 1.0 - np.sum(v[1:]) # i = 0 : no change (phantom jump)
    i = np.searchsorted(np.cumsum(v), np.random.random(), side="right")
    jump = i > 0 # Test if the jump is a true (i > 0) or phantom jump (i == 0)
    if jump:
        state[0, i] += np.random.exponential(1/b[i])

    return U, jump, state

def _step_jit(state: np.ndarray,
              basal: np.ndarray,
              inter: np.ndarray,
              d0: np.ndarray, 
              d1: np.ndarray,
              s1: np.ndarray,
              k0: np.ndarray,
              k1: np.ndarray,
              b: np.ndarray,
              tau: Optional[float]) -> Tuple[float, bool, np.ndarray]:
    """
    Compute the next jump and the next step of the
    thinning method, in the case of the bursty model.
    """
    if tau is None:
        # Adaptive thinning parameter
        tau = np.sum(_kon_bound_jit(state, basal, inter, d0, d1, s1, k0, k1))
    
    # 0. Draw the waiting time before the next jump
    U = np.random.exponential(scale=1/tau)
    
    # 1. Update the continuous states
    state = _flow_jit(U, state, d0, d1, s1)
    
    # 2. Compute the next jump
    v = _kon_jit(state[1], basal, inter, k0, k1)
    #### # Fix precision errors
    # s = np.sum(v[1:])
    # if s > tau:
    #     tau = s
    # v[1:] /= (tau + 1e-10) # i = 1, ..., G-1 : burst of mRNA i
    ####
    v[1:] /= tau # i = 1, ..., G-1 : burst of mRNA i
    v[0] = 1.0 - np.sum(v[1:]) # i = 0 : no change (phantom jump)
    # i = np.nonzero(np.random.multinomial(1, v))[0][0]
    # use this instead of multinomial because of https://github.com/numba/numba/issues/3426
    # https://github.com/numba/numba/issues/2539#issuecomment-507306369
    i = np.searchsorted(np.cumsum(v), np.random.random(), side="right")
    jump = i > 0 # Test if the jump is a true (i > 0) or phantom jump (i == 0)
    if jump:
        state[0, i] += np.random.exponential(1/b[i])
    
    return U, jump, state

def _create_simulation(step, flow):
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
                   tau: Optional[float]) -> Tuple[np.ndarray, int, int]:
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

        # Remove the stimulus
        return states, phantom_jump_count, true_jump_count
    
    return simulation

simulation = _create_simulation(step, flow)
_simulation_jit = None

class BurstyPDMP(Simulation):
    """
    Bursty PDMP version of the network model (promoters not described)
    """

    def __init__(self, 
                 thin_adapt: bool = True, 
                 verbose: bool = False, 
                 use_numba: bool = False) -> None:
        self.thin_adapt : bool  = thin_adapt
        self.is_verbose : bool  = verbose
        self._use_numba, self._simulation = False, simulation
        self.use_numba: bool = use_numba

    @property
    def use_numba(self) -> bool:
        return self._use_numba
    
    @use_numba.setter
    def use_numba(self, use_numba: bool) -> None:
        global _kon_jit, _kon_bound_jit, _flow_jit, _step_jit, _simulation_jit

        if self._use_numba != use_numba:
            if use_numba:
                if _simulation_jit is None:
                    from numba import njit
                    _kon_jit = njit()(_kon_jit)
                    _kon_bound_jit = njit()(_kon_bound_jit)
                    _flow_jit = njit()(flow)
                    _step_jit = njit()(_step_jit)
                    _simulation_jit = njit()(_create_simulation(_step_jit,
                                                                _flow_jit))
                self._simulation = _simulation_jit
            else:
                self._simulation = simulation
            
            self._use_numba = use_numba
    
    def run(self,
            time_points: np.ndarray,
            initial_state: np.ndarray,
            parameter: NetworkParameter) -> Simulation.Result:
        """
        Perform simulation of the network model (bursty PDMP version).
        """
        k0 = parameter.burst_frequency_min
        k1 = parameter.burst_frequency_max

        # Thinning parameter
        tau = None if self.thin_adapt else np.sum(k1)
        
        states, phantom_jump_count, true_jump_count = self._simulation(
            state=initial_state,
            time_points=time_points,
            basal=parameter.basal.filled(),
            inter=parameter.interaction.filled(),
            d0=parameter.degradation_rna.filled(fill_value=1.0),
            d1=parameter.degradation_protein.filled(fill_value=2.0),
            s1=parameter.creation_protein.filled(), 
            k0=k0.filled(), k1=k1.filled(), 
            b=parameter.burst_size_inv.filled(),
            tau=tau
        )
        
        if self.is_verbose:
            # Display info about jumps
            total_jump = phantom_jump_count + true_jump_count
            print(f'Exact simulation used {total_jump} jumps '
                    f'including {phantom_jump_count} phantom jumps '
                    f'({100*phantom_jump_count/total_jump : .2f}%)')
        
        return self.Result(time_points, states[:, 0], states[:, 1])
