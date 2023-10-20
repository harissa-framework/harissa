"""
Core functions for network inference using likelihood maximization
"""
from .hartree import Hartree, p1, grad_p1, infer_proteins
import numpy as np
from scipy.special import psi, expit, gammaln
from scipy.optimize import minimize
from numba import njit

p1 = njit()(p1)
grad_p1 = njit()(grad_p1)

@njit
def penalization(theta, theta0, t, s):
    """
    Penalization of network parameters.
    """
    nb_genes = theta.shape[0]
    p = 0
    for i in range (1, nb_genes):
        # Penalization of basal parameters
        p += 2 * t * p1(theta[i,0]-theta0[i,0], s)
        # Penalization of stimulus parameters
        p += t * p1(theta[0,i]-theta0[0,i], s)
        # Penalization of diagonal parameters
        p += (theta[i,i]-theta0[i,i])**2
        for j in range(1, nb_genes):
            # Penalization of interaction parameters
            p += p1(theta[i,j]-theta0[i,j], s)
            if i < j:
                # Competition between interaction parameters
                p += p1(theta[i,j], s) * p1(theta[j,i], s)
    # Final penalization
    return p

@njit
def grad_penalization(theta, theta0, t, s):
    """
    Penalization gradient of network parameters.
    """
    nb_genes = theta.shape[0]
    gradp = np.zeros((nb_genes, nb_genes))
    for i in range (1, nb_genes):
        # Penalization of basal parameters
        gradp[i,0] += 2 * t * grad_p1(theta[i,0]-theta0[i,0], s)
        # Penalization of stimulus parameters
        gradp[0,i] += t * grad_p1(theta[0,i]-theta0[0,i], s)
        # Penalization of diagonal parameters
        gradp[i,i] += 2*(theta[i,i]-theta0[i,i])
        for j in range(1, nb_genes):
            # Penalization of interaction parameters
            gradp[i,j] += grad_p1(theta[i,j]-theta0[i,j], s)
            if i != j:
                # Competition between interaction parameters
                gradp[i,j] += grad_p1(theta[i,j], s) * p1(theta[j,i], s)
    # Final penalization
    return gradp

def objective(theta, theta0, x, y, a, c, d, l, t, s):
    """
    Objective function to be minimized (one time point).
    """
    C, nb_genes = x.shape
    theta = theta.reshape((nb_genes, nb_genes))
    basal = theta[:,0]
    sigma = expit(basal + y @ theta)[:,1:]
    x, y = x[:,1:], y[:,1:]
    a0, a1, c, d = a[0,1:], a[1,1:], c[1:], d[1:]
    # Compute the log-likelihood
    ay, e = a1*y, a0/a1
    cxi = c * (e + (1-e)*sigma)
    q = (d*ay + gammaln(ay + x) - gammaln(ay) - c*y
        + (cxi-1)*np.log(y) + np.log(c)*cxi - gammaln(cxi))
    return l*penalization(theta, theta0, t, s) - np.sum(q)/C

def grad_theta(theta, theta0, x, y, a, c, d, l, t, s):
    """
    Objective gradient (one time point).
    """
    C, nb_genes = x.shape
    theta = theta.reshape((nb_genes, nb_genes))
    basal = theta[:,0]
    sigma = expit(basal + y @ theta)[:,1:]
    a0, a1, c, d = a[0,1:], a[1,1:], c[1:], d[1:]
    # Pivotal vector u
    e = a0/a1
    xi = e + (1-e)*sigma
    u = c * sigma * (1-xi) * (np.log(c*y[:,1:]) - psi(c*xi))
    # Compute the objective gradient
    dq = np.zeros((nb_genes, nb_genes))
    # Basal parameters
    dq[1:,0] += np.sum(u, axis=0)
    # Interaction parameters
    dq[:,1:] += y.T @ u
    dq = l*grad_penalization(theta, theta0, t, s) - dq/C
    return dq.reshape(nb_genes**2)

def infer_network(x: np.ndarray,
                  y: np.ndarray,
                  a: np.ndarray,
                  c: np.ndarray, 
                  penalization_strength: float, 
                  tolerance: float,
                  smoothing_threshold: float) -> tuple[np.ndarray, int]:
    """
    Network inference procedure.
    """
    nb_genes = x.shape[1]
    times = np.sort(list(set(x[:,0])))
    T = times.size
    # Useful quantities
    k = x[:,0]
    d = np.log(a[2]/(a[2]+1))
    # Initialization
    theta = np.zeros((T, nb_genes, nb_genes))
    theta0 = np.zeros((nb_genes, nb_genes))
    # Optimization parameters
    params = {'method': 'L-BFGS-B'}
    if tolerance is not None: 
        params['tol'] = tolerance
    # Inference routine
    for t, time in enumerate(times):
        res = minimize(objective, 
                       theta0.reshape(nb_genes**2),
                       args=(theta0, 
                             x[k==time], 
                             y[k==time], 
                             a, c, d, 
                             penalization_strength, 
                             t, 
                             smoothing_threshold),
                       jac=grad_theta, 
                       **params)
        if not res.success:
            print(f'Warning: maximization failed (time {t})')
        # Update theta0
        theta0 = res.x.reshape((nb_genes, nb_genes))
        # Store theta at time t
        theta[t] = theta0
    return theta, res.nit

class Hartree_Numba(Hartree):    
    def __init__(self, 
                 penalization_strength : float = 1, 
                 tolerance: float = 1e-5, 
                 max_iteration: int = 100, 
                 verbose: bool = False):
        super().__init__(penalization_strength, 
                         tolerance, 
                         max_iteration, 
                         verbose)


    def run(self, data: np.ndarray) -> Hartree.Result:
        """
        Infers the network model from the data.
        """
        x = data
        # Time points
        times = np.sort(list(set(x[:,0])))
        nb_genes = x.shape[1]

        # Kinetic parameters
        a = self.get_kinetics(data)
        # Concentration parameter
        c = 100 * np.ones(nb_genes)
        # Get protein levels
        y = infer_proteins(x, a)
        # Inference procedure
        theta, nb_iterations = infer_network(x, y, a, c,
                                             self.penalization_strength,
                                             self.tolerance,
                                             self.smoothing_threshold)
        if self.is_verbose: 
            print(f'Fitted theta in {nb_iterations} iterations')
        # Build the results
        basal_time = {time: np.zeros(nb_genes) for time in times}
        inter_time = {time: np.zeros((nb_genes, nb_genes)) for time in times}
        for t, time in enumerate(times):
            basal_time[time][:] = theta[t][:,0]
            inter_time[time][:,1:] = theta[t][:,1:]
        
        res = Hartree.Result(burst_frequency_min=a[0], 
                             burst_frequency_max=a[1], 
                             burst_size=a[2],
                             basal=basal_time[times[-1]],
                             interaction=inter_time[times[-1]])
        res.basal_time = basal_time
        res.interaction_time = inter_time
        res.y = y
        
        return res