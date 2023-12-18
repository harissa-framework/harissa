"""
Main class for network inference and simulation
"""
import numpy as np
from harissa.parameter import NetworkParameter
from harissa.inference import Inference, default_inference
from harissa.simulation import Simulation, default_simulation

# Main class
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
        self._parameter = _check_parameter_setter(value)

    @property
    def inference(self):
        """Inference method of the network model."""
        return self._inference

    @inference.setter
    def inference(self, value):
        self._inference = _check_inference_setter(value)

    @property
    def simulation(self):
        """Simulation method of the network model."""
        return self._simulation

    @simulation.setter
    def simulation(self, value):
        self._simulation = _check_simulation_setter(value)

    # Parameter shortcuts
    # ===================

    @property
    def n_genes(self):
        """Number of genes in the network model, without stimulus."""
        return _check_parameter_specified(self._parameter).n_genes

    @property
    def n_genes_stim(self):
        """Number of genes in the network model, including stimulus."""
        return _check_parameter_specified(self._parameter).n_genes_stim

    @property
    def burst_frequency_min(self):
        """Minimal bursting frequency for each gene (low expression)."""
        return _check_parameter_specified(self._parameter)._burst[0]

    @property
    def burst_frequency_max(self):
        """Maximal bursting frequency for each gene (high expression)."""
        return _check_parameter_specified(self._parameter)._burst[1]

    @property
    def burst_size_inv(self):
        """Inverse of average burst size for each gene."""
        return _check_parameter_specified(self._parameter)._burst[2]

    @property
    def creation_rna(self):
        """mRNA creation rates. Note that in the transcriptional
        bursting regime, s[0] is not identifiable since it aggregates with
        koff (inverse of average ON time) into parameter b = s[0]/koff."""
        return _check_parameter_specified(self._parameter)._creation[0]

    @property
    def creation_protein(self):
        """Protein creation rates."""
        return _check_parameter_specified(self._parameter)._creation[1]

    @property
    def degradation_rna(self):
        """mRNA degradation rates."""
        return _check_parameter_specified(self._parameter)._degradation[0]

    @property
    def degradation_protein(self):
        """Protein degradation rates."""
        return _check_parameter_specified(self._parameter)._degradation[1]

    @property
    def basal(self):
        """Basal expression level for each gene."""
        return _check_parameter_specified(self._parameter)._basal

    @property
    def interaction(self):
        """Interactions between genes."""
        return _check_parameter_specified(self._parameter)._interaction

    # Legacy shortcuts
    # ================

    @property
    def d(self):
        """Degradation kinetics."""
        return _check_parameter_specified(self._parameter)._degradation

    @property
    def a(self):
        """Bursting kinetics (not normalized)."""
        return _check_parameter_specified(self._parameter)._burst

    @property
    def inter(self):
        """Interactions between genes."""
        return _check_parameter_specified(self._parameter)._interaction

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
        _check_parameter_specified(self.parameter)

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

def _check_parameter_setter(arg):
    if isinstance(arg, NetworkParameter):
        return arg
    raise TypeError('parameter should be a NetworkParameter object')

def _check_inference_setter(arg):
    if isinstance(arg, Inference):
        return arg
    raise TypeError('inference should be an Inference object')

def _check_simulation_setter(arg):
    if isinstance(arg, Simulation):
        return arg
    raise TypeError('simulation should be a Simulation object')

def _check_parameter_specified(arg):
    if arg is not None:
        return arg
    raise AttributeError('model parameter is not specified yet')
