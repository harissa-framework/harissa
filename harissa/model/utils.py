import numpy as np
from harissa.model.model import NetworkModel
from harissa.utils.trees import random_tree

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
