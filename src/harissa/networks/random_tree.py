"""
Generate random tree-shaped activation pathways
"""
from typing import Optional
import numpy as np

from harissa.core.parameter import NetworkParameter
from harissa.utils.random_spanning_tree import random_spanning_tree

def random_tree(
        n_genes: int, 
        weight: Optional[np.ndarray] = None, 
        autoactiv: bool = False
    ) -> NetworkParameter:
    """
    Generate a random tree-like network parameter.
    A tree with root 0 is sampled from the ‘weighted-uniform’ distribution,
    where weight[i,j] is the probability weight of link (i) -> (j).
    The matrix `weight` must satisfy 2 conditions:
    
    - weight[1:, 1:] matrix must be irreducible
    - weight[0, 1:] must contain at least 1 nonzero element
    """
    G = n_genes + 1
    if weight is not None:
        if weight.shape != (G, G):
            raise ValueError('Weight must be n_genes+1 by n_genes+1')
    else: 
        weight = np.ones((G, G))
    # Enforcing the proper structure
    weight = weight - np.diag(np.diag(weight))
    weight[:, 0] = 0
    
    # Generate the network
    tree = random_spanning_tree(weight)
    
    basal = np.zeros(G)
    inter = np.zeros((G, G))
    gene_names = np.array(['', *[str(i) for i in range(1, G)]])

    basal[1:] = -5
    for i, targets in enumerate(tree):
        for j in targets:
            inter[i, j] = 10

    if autoactiv:
        for i in range(1, n_genes+1):
            inter[i, i] = 5

    param = NetworkParameter(n_genes, gene_names)
    param.basal[:] = basal
    param.interaction[:] = inter
    
    return param
