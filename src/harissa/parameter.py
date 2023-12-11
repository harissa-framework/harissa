"""
Main class for network parameters
"""
import numpy as np

class NetworkParameter:
    """
    Parameters of a network model.
    """
    def __init__(self, n_genes: int | None = None) -> None:
        """
        Parameters
        ----------
        n_genes
            number of genes
        """
        # Kinetic parameters
        self.burst_frequency_min : np.ndarray | None = None # a[0]
        """Minimal Kon rate (normalized)"""        
        self.burst_frequency_max : np.ndarray | None = None # a[1]
        """Maximal Kon rate (normalized)""" 
        self.burst_size_inv      : np.ndarray | None = None # a[2]
        """Inverse burst size of mRNA"""
        self.creation_rna        : np.ndarray | None = None # s[0]
        """Normalize rna scales"""
        self.creation_protein    : np.ndarray | None = None # s[1]
        """Normalize protein scales"""
        self.degradation_rna     : np.ndarray | None = None # d[0]
        """mRNA degradation rates (per hour)"""
        self.degradation_protein : np.ndarray | None = None # d[1]
        """protein degradation rates (per hour)"""

        # Network parameters
        self.basal       : np.ndarray | None = None # basal
        """basal"""
        self.interaction : np.ndarray | None = None # inter
        """inter"""

        # Default behaviour
        if n_genes is not None:
            G = n_genes + 1 # Genes plus stimulus
            # Default bursting parameters
            self.burst_frequency_min = np.full(G, 0.0)
            self.burst_frequency_max = np.full(G, 2.0)  
            self.burst_size_inv      = np.full(G, 0.02)  

            # Default degradation rates
            self.degradation_rna     = np.log(np.full(G, 2.0)) / 9.0
            self.degradation_protein = np.log(np.full(G, 2.0)) / 46.0

            # Default creation rates
            scale =  self.burst_size_inv / self.burst_frequency_max
            self.creation_rna = self.degradation_rna * scale
            self.creation_protein = self.degradation_protein * scale 
            
            # Default network parameters
            self.basal       = np.zeros(G)
            self.interaction = np.zeros((G,G))

    def check_all_specified(self) -> bool:
        at_least_one_none = False
        for param in self.__dict__.values():
            if param is None:
                at_least_one_none = True
                break
        return not at_least_one_none