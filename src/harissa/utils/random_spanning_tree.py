import numpy as np

def random_step(state, a):
    """
    Make one step of the random walk on the weighted graph defined by a.
    NB: here we construct an in-tree so all directions are reversed.
    """
    p = a[:,state]/np.sum(a[:,state])
    return np.dot(np.arange(p.size), np.random.multinomial(1, p))

def loop_erasure(path):
    """
    Compute the loop erasure of a given path.
    """ 
    i = np.max(np.arange(len(path))*(np.array(path)==path[0]))
    
    if path[i+1] == path[-1]: 
        return [path[0], path[i+1]]
    else: 
        return [path[0]] + loop_erasure(path[i+1:])

def random_spanning_tree(a):
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
            state = random_step(path[-1], a)
            path.append(state)
        path = loop_erasure(path)
        # Append the loop-erased path to the current tree
        for i in range(len(path)-1):
            v.add(path[i])
            r.remove(path[i])
            tree[path[i+1]].append(path[i])
    
    for i in range(n): 
        tree[i].sort()
    
    return tuple([tuple(tree[i]) for i in range(n)])