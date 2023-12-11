"""
Perform simulations using the ODE model
"""
import numpy as np
from harissa.simulation.simulation import Simulation, NetworkParameter
from harissa.simulation.utils import kon 


def _create_step(kon):
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
    
    return step

step = _create_step(kon)

def _create_simulation(step):
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
                   euler_step:float) -> np.ndarray:
        """
        Simulation of the deterministic limit model, which is relevant when
        promoters and mRNA are much faster than proteins.
        1. Nonlinear ODE system involving proteins only
        2. Mean level of mRNA given protein levels
        """
        states = np.empty((time_points.size, *state.shape))
        dt = euler_step
        if time_points.size > 1:
            dt = min(dt, np.min(time_points[1:] - time_points[:-1]))
        t, step_count = 0.0, 0
        # Core loop for simulation and recording
        for i, time_point in enumerate(time_points):
            while t < time_point:
                state = step(state, basal, inter, d0, d1, s1, k0, k1, b, dt)
                t += dt
                step_count += 1
            states[i] = state
        
        # Remove the stimulus
        return states[..., 1:], step_count, dt
    
    return simulation

simulation = _create_simulation(step)
_simulation_jit = None

class ApproxODE(Simulation):
    """
    ODE version of the network model (very rough approximation of the PDMP)
    """
    def __init__(self, verbose: bool = False, use_numba: bool = False) -> None:
        self.is_verbose: bool  = verbose
        self._use_numba, self._simulation = False, simulation 
        self.use_numba: bool = use_numba

    @property
    def use_numba(self) -> bool:
        return self._use_numba
    
    @use_numba.setter
    def use_numba(self, use_numba: bool) -> None:
        global _simulation_jit

        if self._use_numba != use_numba:
            if use_numba:
                if _simulation_jit is None:
                    from numba import njit
                    kon_jit = njit()(kon)
                    step_jit = njit()(_create_step(kon_jit))
                    _simulation_jit = njit()(_create_simulation(step_jit))
                self._simulation = _simulation_jit
            else:
                self._simulation = simulation
            
            self._use_numba = use_numba
    

    def run(self, 
            initial_state: np.ndarray, 
            time_points: np.ndarray, 
            parameter: NetworkParameter) -> Simulation.Result:
        """
        Perform simulation of the network model (ODE version).
        This is the slow-fast limit of the PDMP model, which is only
        relevant when promoters & mRNA are much faster than proteins.
        p: solution of a nonlinear ODE system involving proteins only
        m: mean mRNA levels given protein levels (quasi-steady state)
        """
        states, step_count, dt = self._simulation(
            state=initial_state, 
            time_points=time_points,
            basal=parameter.basal,
            inter=parameter.interaction,
            d0=parameter.degradation_rna,
            d1=parameter.degradation_protein,
            s1=parameter.creation_protein, 
            k0=parameter.burst_frequency_min * parameter.degradation_rna, 
            k1=parameter.burst_frequency_max * parameter.degradation_rna,
            b=parameter.burst_size,
            euler_step=1e-3/np.max(parameter.degradation_protein)
        )
        
        if self.is_verbose:
            # Display info about steps
            if step_count > 0:
                print(f'ODE simulation used {step_count} steps ' 
                      f'(step size = {dt:.5f})')
            else: 
                print('ODE simulation used no step')
        
        return Simulation.Result(time_points, states[:, 0], states[:, 1])