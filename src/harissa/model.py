"""
Main class for network inference and simulation
"""
import numpy as np
from harissa.parameter import NetworkParameter
from harissa.inference import Inference, default_inference
from harissa.simulation import Simulation, default_simulation

class NetworkModel:
    """
    Handle gene regulatory networks within Harissa.
    """
    def __init__(self, parameter=None, inference=None, simulation=None):
        # Set the attributes
        self._parameter = _initialize_parameter(parameter)
        self._inference = _initialize_inference(inference)
        self._simulation = _initialize_simulation(simulation)

    # Properties
    # ==========

    @property
    def parameter(self):
        """Parameters of the network model."""
        return self._parameter

    @parameter.setter
    def parameter(self, value):
        self._parameter = _check_parameter(value)

    @property
    def inference(self):
        """Inference method of the network model."""
        return self._inference

    @inference.setter
    def inference(self, value):
        self._inference = _check_inference(value)

    @property
    def simulation(self):
        """Simulation method of the network model."""
        return self._simulation

    @simulation.setter
    def simulation(self, value):
        self._simulation = _check_simulation(value)

    # Methods
    # =======

    def fit(self, data: np.ndarray) -> Inference.Result:
        """
        Fit the network model to the data.
        """
        res = self.inference.run(data)
        self.parameter = res.parameter
        return res

    def simulate(self, time_points: np.ndarray, *,
                 M0: np.ndarray | None = None,
                 P0: np.ndarray | None = None,
                 burn_in: float | None = None) -> Simulation.Result:
        """
        Perform simulation of the network model.
        """
        if self.parameter is None:
            raise AttributeError('parameter not specified yet')
        
        t_pts_nb_dim = time_points.ndim 
        if t_pts_nb_dim == 0:
            time_points = np.array([time_points])
        elif t_pts_nb_dim >= 2:
            raise ValueError(f'Time points is a {t_pts_nb_dim}D np.ndarray. '
                              'It must be a 0D or 1D np.ndarray.')
        
        if np.any(time_points != np.sort(time_points)):
            raise ValueError('Time points must appear in increasing order')
        
        # Initial state: row 0 <-> rna, row 1 <-> protein
        initial_state = np.zeros((2, self.parameter.basal.size))

        if M0 is not None:
            initial_state[0, 1:] = M0[1:]
        if P0 is not None: 
            initial_state[1, 1:] = P0[1:]

        # Burn_in simulation without stimulus
        if burn_in is not None:
            res = self.simulation.run(
                initial_state, 
                np.array([burn_in]), 
                self.parameter
            )
            # Update initial state 
            initial_state[0, 1:] = res.rna_levels[-1] 
            initial_state[1, 1:] = res.protein_levels[-1]
            
        # Activate the stimulus
        initial_state[1, 0] = 1.0

        # Final simulation with stimulus
        res = self.simulation.run(initial_state, time_points, self.parameter)

        # NOTE: maybe wrap it to AnnData
        return res


# Utility functions
# =================

def _initialize_parameter(arg):
    # Option 1: NetworkParameter object or None
    if isinstance(arg, NetworkParameter) or (arg is None):
        return arg
    # Option 2: number of genes
    elif isinstance(arg, int):
        return NetworkParameter(arg)
    raise TypeError('parameter argument is not valid')

def _initialize_inference(arg):
    # Option 1: Inference object
    if isinstance(arg, Inference):
        return arg
    # Option 2: default method
    elif arg is None:
        return default_inference()
    raise TypeError('inference argument is not valid')

def _initialize_simulation(arg):
    # Option 1: Simulation object
    if isinstance(arg, Simulation):
        return arg
    # Option 2: default method
    elif arg is None:
        return default_simulation()
    raise TypeError('simulation argument is not valid')

def _check_parameter(arg):
    if isinstance(arg, NetworkParameter):
        return arg
    raise TypeError('parameter should be a NetworkParameter object')

def _check_inference(arg):
    if isinstance(arg, Inference):
        return arg
    raise TypeError('inference should be an Inference object')

def _check_simulation(arg):
    if isinstance(arg, Simulation):
        return arg
    raise TypeError('simulation should be a Simulation object')
