"""
Generate random tree-shaped activation pathways
"""
import numpy as np
from harissa.core.parameter import NetworkParameter

def _random_step(state, a):
    """
    Make one step of the random walk on the weighted graph defined by a.
    NB: here we construct an in-tree so all directions are reversed.
    """
    p = a[:,state]/np.sum(a[:,state])
    return np.dot(np.arange(p.size), np.random.multinomial(1, p))

def _loop_erasure(path):
    """
    Compute the loop erasure of a given path.
    """ 
    i = np.max(np.arange(len(path))*(np.array(path)==path[0]))
    
    if path[i+1] == path[-1]: 
        return [path[0], path[i+1]]
    else: 
        return [path[0]] + _loop_erasure(path[i+1:])

def _random_spanning_tree(a):
    """
    Generate a random spanning tree rooted in node 0 from the uniform
    distribution with weights given by matrix a (using Wilson's method).
    """
    n = a[0].size
    tree = [[] for i in range(n)]
    v = {0} # Vertices of the tree
    r = list(range(1,n)) # Remaining vertices
    
    while len(r) > 0:
        state = r[0]
        path = [state]
        # compute a random path that reaches the current tree
        while path[-1] not in v:
            state = _random_step(path[-1], a)
            path.append(state)
        path = _loop_erasure(path)
        # Append the loop-erased path to the current tree
        for i in range(len(path)-1):
            v.add(path[i])
            r.remove(path[i])
            tree[path[i+1]].append(path[i])
    
    for i in range(n): 
        tree[i].sort()
    
    return tuple([tuple(tree[i]) for i in range(n)])

def random_tree(
        n_genes: int, 
        weight: np.ndarray | None = None, 
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
    tree = _random_spanning_tree(weight)
    basal = np.zeros(G)
    inter = np.zeros((G, G))
    basal[1:] = -5
    for i, targets in enumerate(tree):
        for j in targets:
            inter[i, j] = 10

    if autoactiv:
        for i in range(1, n_genes+1):
            inter[i, i] = 5

    param = NetworkParameter(n_genes)
    param.basal[:] = basal
    param.interaction[:] = inter
    
    return param
