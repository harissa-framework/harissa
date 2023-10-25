"""
Main class for network inference and simulation
"""
import numpy as np
from ..inference.hartree.hartree import Inference, Hartree
from ..simulation.bursty_pdmp.bursty_pdmp import Simulation, BurstyPDMP

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

    def fit(self, data: np.ndarray):
        """
        Fit the network model to the data.
        """
        res : Inference.Result = self.inference.run(data)
        for key, value in res.__dict__.items():
            if hasattr(self, key) and value is not None:
                setattr(self, key, value)
        return res

    def simulate(self, time_points: np.ndarray):
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
        
        nb_dim = time_points.ndim 
        if nb_dim == 0:
            time_points = np.array([time_points])
        elif nb_dim >= 2:
            raise ValueError(f'Time points is a {nb_dim}D np.ndarray. '
                              'It must be a 0D or 1D np.ndarray.')
        
        if np.any(time_points != np.sort(time_points)):
            raise ValueError('Time points must appear in increasing order')

        res : Simulation.Result = self.simulation.run(time_points,
                                                      self.burst_frequency_min,
                                                      self.burst_frequency_max,
                                                      self.burst_size,
                                                      self.degradation_rna,
                                                      self.degradation_protein,
                                                      self.basal,
                                                      self.interaction)       
        return res
