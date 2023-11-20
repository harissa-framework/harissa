"""
Main class for network inference and simulation
"""
import numpy as np
from harissa.inference.hartree.hartree import Inference, Hartree
from harissa.simulation.bursty_pdmp.bursty_pdmp import Simulation, BurstyPDMP
from harissa.utils.trees import random_tree

class NetworkModel:
    """
    Handle networks within Harissa.
    """
    def __init__(self, n_genes: int | None = None, *, 
                 inference: Inference = Hartree(), 
                 simulation: Simulation = BurstyPDMP()):
        # Kinetic parameters
        self.burst_frequency_min : np.ndarray | None = None # Minimal Kon rate (normalized)
        self.burst_frequency_max : np.ndarray | None = None # Maximal Kon rate (normalized)
        self.burst_size          : np.ndarray | None = None # Inverse burst size of mRNA
        self.creation_rna        : np.ndarray | None = None
        self.creation_protein    : np.ndarray | None = None
        self.degradation_rna     : np.ndarray | None = None # mRNA degradation rates (per hour)
        self.degradation_protein : np.ndarray | None = None # protein degradation rates (per hour)
        
        # Network parameters
        self.basal       : np.ndarray | None = None
        self.interaction : np.ndarray | None = None

        # Inference and Simulation parameters
        self.inference  : Inference  = inference
        self.simulation : Simulation = simulation
        
        # Default behaviour
        if n_genes is not None:
            G = n_genes + 1 # Genes plus stimulus
            # Default bursting parameters
            self.burst_frequency_min = np.full(G, 0.0)
            self.burst_frequency_max = np.full(G, 2.0)  
            self.burst_size          = self.burst_frequency_max / 100

            # Default degradation rates
            self.degradation_rna     = np.log(self.burst_frequency_max) / 9.0
            self.degradation_protein = np.log(self.burst_frequency_max) / 46.0 
            
            # Default network parameters
            self.basal       = np.zeros(G)
            self.interaction = np.zeros((G,G))

    def fit(self, data: np.ndarray) -> Inference.Result:
        """
        Fit the network model to the data.
        """
        res = self.inference.run(data)
        for key, value in res.__dict__.items():
            if hasattr(self, key) and value is not None:
                setattr(self, key, value)

        return res

    def simulate(self, time_points: np.ndarray, *,
                 M0: np.ndarray | None = None,
                 P0: np.ndarray | None = None,
                 burn_in: float | None = None) -> Simulation.Result:
        """
        Perform simulation of the network model.
        """
        # Check parameters
        if (self.burst_frequency_min is None 
            or self.burst_frequency_max is None 
            or self.burst_size is None 
            or self.degradation_rna is None 
            or self.degradation_protein is None 
            or self.basal is None
            or self.interaction is None):
            raise ValueError('Model parameters not yet specified')
        
        param = Simulation.Parameter(self.burst_frequency_min,
                                     self.burst_frequency_max,
                                     self.burst_size,
                                     self.degradation_rna,
                                     self.degradation_protein,
                                     self.basal,
                                     self.interaction)
        
        t_pts_nb_dim = time_points.ndim 
        if t_pts_nb_dim == 0:
            time_points = np.array([time_points])
        elif t_pts_nb_dim >= 2:
            raise ValueError(f'Time points is a {t_pts_nb_dim}D np.ndarray. '
                              'It must be a 0D or 1D np.ndarray.')
        
        if np.any(time_points != np.sort(time_points)):
            raise ValueError('Time points must appear in increasing order')
        
        # Initial state: row 0 <-> rna, row 1 <-> protein
        param.initial_state = np.zeros((2, self.basal.size))

        if M0 is not None:
            param.initial_state[0, 1:] = M0[1:]
        if P0 is not None: 
            param.initial_state[1, 1:] = P0[1:]

        # Burn_in simulation without stimulus
        if burn_in is not None:
            param.time_points = np.array([burn_in])
            res = self.simulation.run(param)
            # Update initial state 
            param.initial_state[0, 1:] = res.rna_levels[-1] 
            param.initial_state[1, 1:] = res.protein_levels[-1]
            
        # Activate the stimulus
        param.initial_state[1, 0] = 1.0

        param.time_points = time_points

        # Final simulation with stimulus
        res = self.simulation.run(param)

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

    model = NetworkModel(n_genes)
    model.basal = basal
    model.interaction = inter
    
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

    model = NetworkModel(n_genes)
    model.basal = basal
    model.interaction = inter
    
    return model
