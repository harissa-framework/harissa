"""
Generate cascade networks
"""
import numpy as np
from harissa.models.network_model import NetworkModel

def cascade(n_genes):
    """
    Generate a simple activation cascade (1) -> (2) -> ... -> (n_genes).
    """
    G = n_genes + 1
    basal = np.zeros(G)
    inter = np.zeros((G,G))
    basal[1:] = -5 # Low basal level of downstream genes
    for i in range(n_genes):
        inter[i,i+1] = 10
    return basal, inter

class Cascade(NetworkModel):
    """
    Particular network with a cascade structure.
    """
    def __init__(self, n_genes, autoactiv=False):
        # Get NetworkModel default features
        super().__init__(n_genes)
        # New network parameters
        basal, inter = cascade(n_genes)
        if autoactiv:
            for i in range(1, n_genes+1):
                inter[i,i] = 5
        self.basal = basal
        self.inter = inter
