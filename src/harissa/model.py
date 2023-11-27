"""
Main class for network inference and simulation
"""
import numpy as np
from harissa.parameter import NetworkParameter
from harissa.inference.hartree.hartree import Inference, Hartree
from harissa.simulation.bursty_pdmp.bursty_pdmp import Simulation, BurstyPDMP
from harissa.utils.trees import random_tree

class NetworkModel:
    """
    Handle networks within Harissa.
    """
    def __init__(self, parameter: NetworkParameter = NetworkParameter(), *, 
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
        if res.parameter.check_all_specified:
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
    

def cascade(n_genes, autoactiv=False):
    """
    Generate a simple activation cascade (1) -> (2) -> ... -> (n_genes).
    """
    G = n_genes + 1
    basal = np.zeros(G)
    inter = np.zeros((G, G))
    basal[1:] = -5 # Low basal level of downstream genes
    for i in range(n_genes):
        inter[i, i+1] = 10

    if autoactiv:
        for i in range(1, n_genes+1):
            inter[i, i] = 5

    model = NetworkModel(NetworkParameter(n_genes))
    model.parameter.basal = basal
    model.parameter.interaction = inter
    
    return model

def tree(n_genes, weight=None, autoactiv=False):
    """
    Generate a random tree-like network model.
    A tree with root 0 is sampled from the ‘weighted-uniform’ distribution,
    where weight[i,j] is the probability weight of link (i) -> (j).
    """
    G = n_genes + 1
    if weight is not None:
        if weight.shape != (G, G):
            raise ValueError('Weight must be n_genes+1 by n_genes+1')
    else: 
        weight = np.ones((G, G))
    # Enforcing the proper structure
    weight[:, 0] = 0
    weight = weight - np.diag(np.diag(weight))
    # Generate the network
    tree = random_tree(weight)
    basal = np.zeros(G)
    inter = np.zeros((G, G))
    basal[1:] = -5
    for i, targets in enumerate(tree):
        for j in targets:
            inter[i, j] = 10

    if autoactiv:
        for i in range(1,n_genes+1):
            inter[i, i] = 5

    model = NetworkModel(NetworkParameter(n_genes))
    model.parameter.basal = basal
    model.parameter.interaction = inter
    
    return model
