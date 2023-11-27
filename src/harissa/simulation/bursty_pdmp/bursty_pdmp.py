"""
Perform simulations using the PDMP model
"""
import numpy as np
from harissa.simulation.simulation import Simulation, NetworkParameter
from harissa.simulation.utils import kon, kon_bound, flow

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

_kon_jit = None
_kon_bound_jit = None
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

def _step_jit(state: np.ndarray,
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
        tau = np.sum(_kon_bound_jit(state, basal, inter, d0, d1, s1, k0, k1))
    jump = False # Test if the jump is a true or phantom jump
    
    # 0. Draw the waiting time before the next jump
    U = np.random.exponential(scale=1/tau)
    
    # 1. Update the continuous states
    state = _flow_jit(U, state, d0, d1, s1)
    
    # 2. Compute the next jump
    v = _kon_jit(state[1], basal, inter, k0, k1) / tau # i = 1, ..., G-1 : burst of mRNA i
    v[0] = 1 - np.sum(v[1:]) # i = 0 : no change (phantom jump)
    i = np.nonzero(np.random.multinomial(1, v))[0][0]
    if i > 0:
        state[0, i] += np.random.exponential(1/b[i])
        jump = True
    
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

        # Remove the stimulus
        return states[..., 1:], phantom_jump_count, true_jump_count
    
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
                    _kon_jit = njit()(kon)
                    _kon_bound_jit = njit()(kon_bound)
                    _flow_jit = njit()(flow)
                    _step_jit = njit()(_step_jit)
                    _simulation_jit = njit()(_create_simulation(_step_jit, 
                                                                _flow_jit))
                self._simulation = _simulation_jit
            else:
                self._simulation = simulation
            
            self._use_numba = use_numba
    
    def run(self, 
            initial_state: np.ndarray, 
            time_points: np.ndarray, 
            parameter: NetworkParameter) -> Simulation.Result:
        """
        Perform simulation of the network model (bursty PDMP version).
        """
        k0 = parameter.burst_frequency_min * parameter.degradation_rna
        k1 = parameter.burst_frequency_max * parameter.degradation_rna

        # Normalize protein scales
        s1 = (parameter.degradation_protein * parameter.burst_size 
              / parameter.burst_frequency_max)

        # Thinning parameter
        tau = None if self.thin_adapt else np.sum(k1[1:])
        
        states, phantom_jump_count, true_jump_count = self._simulation(
            state=initial_state,
            time_points=time_points,
            basal=parameter.basal,
            inter=parameter.interaction,
            d0=parameter.degradation_rna,
            d1=parameter.degradation_protein,
            s1=s1, k0=k0, k1=k1, b=parameter.burst_size,
            tau=tau)
        
        if self.is_verbose:
            # Display info about jumps
            total_jump = phantom_jump_count + true_jump_count
            if total_jump > 0:
                print(f'Exact simulation used {total_jump} jumps ' 
                      f'including {phantom_jump_count} phantom jumps ' 
                      f'({100*phantom_jump_count/total_jump : .2f}%)')
            else: 
                print('Exact simulation used no jump')
        
        return Simulation.Result(time_points, states[:, 0], states[:, 1])