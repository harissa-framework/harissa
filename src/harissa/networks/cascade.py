"""
Generate activation cascade networks
"""
import numpy as np
from harissa.core.parameter import NetworkParameter

def cascade(n_genes: int, autoactiv: bool = False) -> NetworkParameter:
    """
    Generate a simple activation cascade (1) -> (2) -> ... -> (n_genes).
    """
    G = n_genes + 1
    basal = np.zeros(G)
    inter = np.zeros((G, G))
    gene_names = np.array(['', *[str(i) for i in range(1, G)]])

    basal[1:] = -5 # Low basal level of downstream genes 
    for i in range(n_genes):
        inter[i, i+1] = 10

    if autoactiv:
        for i in range(1, n_genes+1):
            inter[i, i] = 5

    param = NetworkParameter(n_genes, gene_names)
    param.basal[:] = basal
    param.interaction[:] = inter
    
    return param
