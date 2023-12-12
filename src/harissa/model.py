"""
Main class for network inference and simulation
"""
import numpy as np
from harissa.parameter import NetworkParameter
from harissa.inference.hartree.hartree import Inference, Hartree
from harissa.simulation.bursty_pdmp.bursty_pdmp import Simulation, BurstyPDMP

class NetworkModel:
    """
    Handle networks within Harissa.
    """
    def __init__(self, parameter: NetworkParameter = None, *, 
                 inference: Inference = Hartree(), 
                 simulation: Simulation = BurstyPDMP()):
        self.parameter : NetworkParameter = parameter
        self.inference : Inference = inference
        self.simulation : Simulation = simulation

    def fit(self, data: np.ndarray) -> Inference.Result:
        """
        Fit the network model to the data.
        """
        res = self.inference.run(data)
        if res.parameter.check_all_specified():
            self.parameter = res.parameter

        return res

    def simulate(self, time_points: np.ndarray, *,
                 M0: np.ndarray | None = None,
                 P0: np.ndarray | None = None,
                 burn_in: float | None = None) -> Simulation.Result:
        """
        Perform simulation of the network model.
        """
        if not self.parameter.check_all_specified():
            raise ValueError('Model parameters not yet specified')
        
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

        time_points = time_points

        # Final simulation with stimulus
        res = self.simulation.run(initial_state, time_points, self.parameter)

        #NOTE maybe wrap it to AnnData
                   
        return res
    