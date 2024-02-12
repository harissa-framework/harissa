"""
Main class for network inference and simulation
"""
import numpy as np
from harissa.core.parameter import NetworkParameter
from harissa.core.inference import Inference
from harissa.core.simulation import Simulation
from harissa.inference import default_inference
from harissa.simulation import default_simulation


# Main class
class NetworkModel:
    """
    Handle gene regulatory networks within Harissa.
    """

    def __init__(
        self,
        parameter=None,
        inference=default_inference(),
        simulation=default_simulation(),
    ):
        # Option 1: NetworkParameter object or None
        if isinstance(parameter, NetworkParameter) or (parameter is None):
            self._parameter = parameter
        # Option 2: number of genes
        elif isinstance(parameter, int):
            self._parameter = NetworkParameter(parameter)
        else:
            raise TypeError(
                "parameter argument must be "
                "an int or a NetworkParameter (or None)."
            )

        self._inference = _check_type(inference, Inference)
        self._simulation = _check_type(simulation, Simulation)

    # Properties
    # ==========

    @property
    def parameter(self):
        """Parameters of the network model."""
        return self._parameter

    @parameter.setter
    def parameter(self, value):
        self._parameter = _check_type(value, NetworkParameter)

    @property
    def inference(self):
        """Inference method of the network model."""
        return self._inference

    @inference.setter
    def inference(self, value):
        self._inference = _check_type(value, Inference)

    @property
    def simulation(self):
        """Simulation method of the network model."""
        return self._simulation

    @simulation.setter
    def simulation(self, value):
        self._simulation = _check_type(value, Simulation)

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
        return _check_parameter_specified(self._parameter).burst_frequency_min

    @property
    def burst_frequency_max(self):
        """Maximal bursting frequency for each gene (high expression)."""
        return _check_parameter_specified(self._parameter).burst_frequency_max

    @property
    def burst_size_inv(self):
        """Inverse of average burst size for each gene."""
        return _check_parameter_specified(self._parameter).burst_size_inv

    @property
    def creation_rna(self):
        """mRNA creation rates. Note that in the transcriptional
        bursting regime, s[0] is not identifiable since it aggregates with
        koff (inverse of average ON time) into parameter b = s[0]/koff."""
        return _check_parameter_specified(self._parameter).creation_rna

    @property
    def creation_protein(self):
        """Protein creation rates."""
        return _check_parameter_specified(self._parameter).creation_protein

    @property
    def degradation_rna(self):
        """mRNA degradation rates."""
        return _check_parameter_specified(self._parameter).degradation_rna

    @property
    def degradation_protein(self):
        """Protein degradation rates."""
        return _check_parameter_specified(self._parameter).degradation_protein

    @property
    def basal(self):
        """Basal expression level for each gene."""
        return _check_parameter_specified(self._parameter).basal

    @property
    def interaction(self):
        """Interactions between genes."""
        return _check_parameter_specified(self._parameter).interaction

    # Legacy shortcuts
    # ================

    @property
    def d(self):
        """Degradation kinetics."""
        return _check_parameter_specified(self._parameter).d

    @property
    def a(self):
        """Bursting kinetics (not normalized)."""
        return _check_parameter_specified(self._parameter).a

    @property
    def inter(self):
        """Interactions between genes."""
        return _check_parameter_specified(self._parameter).interaction

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
                 initial_state: np.ndarray | None = None,
                 burn_in: float | None = None) -> Simulation.Result:
        """
        Perform simulation of the network model.
        Note: the stimulus is given by initial_state[1,0] (protein 0).
        If burn_in is not None, this value is ignored.
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
        if initial_state is None:
            initial_state = np.zeros((2, self.parameter.basal.size))
            # Activate the stimulus
            initial_state[1, 0] = 1.0
        else:
            initial_state = initial_state.copy()

        # Burn_in simulation without stimulus
        if burn_in is not None:
            # Deactivate the stimulus
            initial_state[1, 0] = 0.0
            # Burn-in simulation
            res_burn_in = self.simulation.run(
                initial_state,
                np.array([burn_in]),
                self.parameter
            )
            # Update initial state 
            initial_state = res_burn_in.final_state
            # Activate the stimulus
            initial_state[1, 0] = 1.0

        # Main simulation
        res = self.simulation.run(initial_state, time_points, self.parameter)

        # NOTE: maybe wrap it to AnnData
        return res

    # FEATURE: dynamic stimulus
    # def simulate_dynamic(self, time_points, stimulus_states):
    #     pass

# Utility functions
# =================
def _check_type(arg, cls):
    if isinstance(arg, cls):
        return arg
    raise TypeError(f'argument of type {type(arg).__name__} '
                    f'should be a {cls.__name__} object.')

def _check_parameter_specified(arg):
    if arg is not None:
        return arg
    raise AttributeError('model parameter is not specified yet')
