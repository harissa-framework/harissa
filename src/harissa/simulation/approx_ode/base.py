"""
Perform simulations using the ODE model
"""
import numpy as np
from harissa.core.parameter import NetworkParameter
from harissa.core.simulation import Simulation
from harissa.simulation.approx_ode.utils import kon

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
        return states, step_count, dt
    
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
        global _kon_jit, _simulation_jit

        if self._use_numba != use_numba:
            if use_numba:
                if _simulation_jit is None:
                    from numba import njit
                    _kon_jit = njit()(_kon_jit)
                    step_jit = njit()(_create_step(_kon_jit))
                    _simulation_jit = njit()(_create_simulation(step_jit))
                self._simulation = _simulation_jit
            else:
                self._simulation = simulation
            
            self._use_numba = use_numba
    

    def run(self, 
            time_points: np.ndarray,
            initial_state: np.ndarray, 
            parameter: NetworkParameter) -> Simulation.Result:
        """
        Perform simulation of the network model (ODE version).
        This is the slow-fast limit of the PDMP model, which is only
        relevant when promoters & mRNA are much faster than proteins.
        p: solution of a nonlinear ODE system involving proteins only
        m: mean mRNA levels given protein levels (quasi-steady state)
        """
        k0 = parameter.burst_frequency_min
        k1 = parameter.burst_frequency_max

        states, step_count, dt = self._simulation(
            state=initial_state, 
            time_points=time_points,
            basal=parameter.basal.filled(),
            inter=parameter.interaction.filled(),
            d0=parameter.degradation_rna.filled(fill_value=1.0),
            d1=parameter.degradation_protein.filled(),
            s1=parameter.creation_protein.filled(),
            k0=k0.filled(), k1=k1.filled(),
            b=parameter.burst_size_inv.filled(fill_value=1.0),
            euler_step=1e-3/np.max(parameter.degradation_protein)
        )
        
        if self.is_verbose:
            # Display info about steps
            print(f'ODE simulation used {step_count} steps '
                    f'(step size = {dt:.5f})')
            
        return Simulation.Result(time_points, states[:, 0], states[:, 1])
