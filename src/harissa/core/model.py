"""
Main class for network inference and simulation
"""
from typing import List, Tuple, Union, Optional
from harissa.core.dataset import Dataset
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
        inference=None,
        simulation=None,
    ):
        # Option 1: NetworkParameter object or None
        if isinstance(parameter, NetworkParameter) or (parameter is None):
            self._parameter = parameter
        # Option 2: number of genes
        elif isinstance(parameter, int):
            self._parameter = NetworkParameter(parameter)
        else:
            raise TypeError(('parameter argument must be '
                'an int or a NetworkParameter (or None).'))

        self.inference = inference or default_inference()
        self.simulation = simulation or default_simulation()

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

    def fit(self, data: Dataset) -> Inference.Result:
        """
        Fit the network model to the data.
        """
        if not isinstance(data, Dataset):
            raise TypeError(( 'data must be Dataset objet ' 
                             f'and not a(n) {type(data)}.'))
        
        if self.parameter is None:
            param = NetworkParameter(
                data.count_matrix.shape[1] - 1,
                data.gene_names
            )
        else:
            param = self.parameter.copy()
        
        res = self.inference.run(data, param)
        self.parameter = res.parameter
        return res

    def simulate(self,
        time_points: np.ndarray,
        initial_state: Optional[np.ndarray] = None,
        initial_time: float = 0.0
        ) -> Simulation.Result:
        """
        Perform simulation of the network model.
        Note: the stimulus is given by initial_state[1,0] (protein 0).

        Note: time points must be in increasing order, and the first
        time point must be greater than or equal to `initial_time`.
        """
        parameter = _check_parameter_specified(self.parameter)

        if not (isinstance(time_points, np.ndarray) 
                or isinstance(time_points, np.generic)):
            raise TypeError(('time_points must be a np.ndarray and '
                             f'not a(n) {type(time_points)}'))

        t_pts_nb_dim = time_points.ndim
        if t_pts_nb_dim == 0:
            time_points = np.array([time_points])
        elif t_pts_nb_dim >= 2:
            raise ValueError((f'Time points is a {t_pts_nb_dim}D np.ndarray. '
                               'It must be a 0D or 1D np.ndarray.'))
        if not np.array_equal(time_points, np.unique(time_points)):
            raise ValueError('Time points must appear in increasing order')
        if time_points[0] < initial_time:
            raise ValueError(('The first time point must be greater than or '
                              'equal to initial time.'))

        # Initial state: row 0 <-> rna, row 1 <-> protein
        state_shape = (2, self.n_genes_stim)
        if initial_state is None:
            initial_state = np.zeros(state_shape)
            # Activate the stimulus
            initial_state[1, 0] = 1.0
        else:
            if (not isinstance(initial_state, np.ndarray) 
                or initial_state.shape != state_shape):
                raise TypeError(('initial_state must be a 2D np.ndarray '
                                f'of shape {state_shape}.'))
            initial_state = initial_state.copy()

        # Main simulation
        res = self.simulation.run(
            time_points - initial_time,
            initial_state,
            parameter
        )
        # Set time points
        res.time_points[:] = time_points

        # NOTE: maybe wrap it to AnnData
        return res
    
    def burn_in(self, duration: float) -> np.ndarray:
        """
        Burn-in simulation without stimulus

        Parameters
        ----------
        duration :
            Simulation duration

        Returns
        -------
            Simulation final state with stimulus activated
        """
        # Burn-in simulation
        res_burn_in = self.simulate(
            np.array([duration]),
            np.zeros((2, self.n_genes_stim))
        )
        
        final_state = res_burn_in.final_state
        # Activate the stimulus
        final_state[1, 0] = 1.0

        return final_state

    
    def simulate_dataset(self,
            time_points: np.ndarray, 
            n_cells: Union[int, List[int], Tuple[int], np.ndarray],
            burn_in_duration: float = 5.0
        ) -> Dataset:
        """
        Generate a dataset

        Parameters
        ----------
        time_points:
            The time points
        n_cells:
            The number of cells per time point 

        Returns
        -------
        Dataset
            The simulated dataset
        """
        if not isinstance(time_points, np.ndarray) or time_points.ndim != 1:
            raise TypeError(('time_points must be a 1D np.ndarray '
                            f'and not {type(time_points)}'))

        if isinstance(n_cells, int):
            n_cells = np.full(time_points.size, n_cells, dtype=np.int_)
        elif isinstance(n_cells, (list, tuple)):
            n_cells = np.array(n_cells, dtype=np.int_)
        elif (not isinstance(n_cells, np.ndarray) 
              or n_cells.dtype != np.int_
              or n_cells.ndim != 1):
            raise TypeError(('n_cells must be an int 1D np.ndarray '
                            f' and not {type(n_cells)}'))

        if n_cells.size != time_points.size:
            raise ValueError((f'n_cells ({n_cells.size}) must have the same '
                              f'size as time_points ({time_points.size})'))
        
        if np.any(time_points < 0):
            raise ValueError(('time_points must contains '
                              'only non negative elements.'))

        if np.any(n_cells <= 0):
            raise ValueError('n_cells must contains only positive elements.')
        
        tot_cells = np.sum(n_cells)

        cells_time = np.empty(tot_cells)
        count_matrix = np.empty((tot_cells, self.n_genes_stim), dtype=np.uint)
        offset = 0
        
        for i in range(time_points.size):
            time = time_points[i]
            n_cell = n_cells[i]

            # Copy time points
            cells_time[offset:offset+n_cell] = time
            # Stimulus
            count_matrix[offset:offset+n_cell, 0] = time != 0.0
            # Generate data
            for cell in range(n_cell):
                count_matrix[offset + cell, 1:] = np.random.poisson(
                    self.simulate(
                        time,
                        initial_state=self.burn_in(burn_in_duration)
                    ).rna_levels[0, 1:]
                )

            offset += n_cell
        
        return Dataset(cells_time, count_matrix)

    # FEATURE: dynamic stimulus
    # def simulate_dynamic(self, time_points, stimulus_states):
    #     pass

# Utility functions
# =================
def _check_type(arg, cls):
    if isinstance(arg, cls):
        return arg
    raise TypeError((f'argument of type {type(arg).__name__} '
                    f'should be a {cls.__name__} object.'))

def _check_parameter_specified(arg):
    if arg is not None:
        return arg
    raise AttributeError('model parameter is not specified yet')
