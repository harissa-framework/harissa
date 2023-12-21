"""
Core functions for network inference using likelihood maximization
"""
import numpy as np
from scipy.special import psi, polygamma, expit, gammaln
from scipy.optimize import minimize
from harissa.parameter import NetworkParameter
from harissa.inference import Inference
from harissa.inference.hartree.utils import estimate_gamma_poisson

def p1(x, s):
    """
    Smoothed L1 penalization.
    """
    return (x-s/2)*(x>s) - (x+s/2)*(-x>s) + ((x**2)/(2*s))*(x<=s and -x<=s)

def grad_p1(x, s):
    """
    Smoothed L1 penalization gradient.
    """
    return 1*(x>s) - 1*(-x>s) + (x/s)*(x<=s and -x<=s)

def _create_penalization(p1):
    def penalization(theta, theta0, t, s):
        """
        Penalization of network parameters.
        """
        nb_genes = theta.shape[0]
        p = 0
        for i in range(1, nb_genes):
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
    
    return penalization 

penalization = _create_penalization(p1)

def _create_grad_penalization(p1, grad_p1):
    def grad_penalization(theta, theta0, t, s):
        """
        Penalization gradient of network parameters.
        """
        nb_genes = theta.shape[0]
        grad_p = np.zeros((nb_genes, nb_genes))
        for i in range (1, nb_genes):
            # Penalization of basal parameters
            grad_p[i,0] += 2 * t * grad_p1(theta[i,0]-theta0[i,0], s)
            # Penalization of stimulus parameters
            grad_p[0,i] += t * grad_p1(theta[0,i]-theta0[0,i], s)
            # Penalization of diagonal parameters
            grad_p[i,i] += 2*(theta[i,i]-theta0[i,i])
            for j in range(1, nb_genes):
                # Penalization of interaction parameters
                grad_p[i,j] += grad_p1(theta[i,j]-theta0[i,j], s)
                if i != j:
                    # Competition between interaction parameters
                    grad_p[i,j] += grad_p1(theta[i,j], s) * p1(theta[j,i], s)
        # Final penalization
        return grad_p
    
    return grad_penalization

grad_penalization = _create_grad_penalization(p1, grad_p1)

def _create_objective(penalization):
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
    
    return objective

objective = _create_objective(penalization)

def _create_grad_theta(grad_penalization):
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
    
    return grad_theta

grad_theta = _create_grad_theta(grad_penalization)

def infer_kinetics(x: np.ndarray, 
                   times: np.ndarray, 
                   tolerance: float, 
                   max_iteration: int) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Infer parameters a[0], ..., a[m-1] and b of a Gamma-Poisson 
    model with time-dependant a and constant b 
    for a given gene at m time points.

    Parameters
    ----------
    x : ndarray
        `x[k]` is the gene expression of cell `k`
    times : ndarray
        `times[k]` is the time point of cell `k`
    """
    t = np.sort(list(set(times)))
    m = t.size
    n = np.zeros(m) # Number of cells for each time point
    a = np.zeros(m)
    b = np.zeros(m)
    # Initialization of a and b
    for i in range(m):
        cells = (times == t[i])
        n[i] = np.sum(cells)
        a[i], b[i] = estimate_gamma_poisson(x[cells])
    b = np.mean(b)
    # Newton-like method
    k, c = 0, 0
    sx = np.sum(x)
    while (k == 0) or (k < max_iteration and c > tolerance):
        da = np.zeros(m)
        for i in range(m):
            if a[i] > 0:
                cells = (times == t[i])
                z = a[i] + x[cells]
                p0 = np.sum(psi(z))
                p1 = np.sum(polygamma(1, z))
                d = n[i]*(np.log(b)-np.log(b+1)-psi(a[i])) + p0
                h = p1 - n[i]*polygamma(1, a[i])
                da[i] = -d/h
        anew = a + da
        if np.sum(anew < 0) == 0: 
            a[:] = anew
        else:
            max_test = 5
            test = 0
            da *= 0.5
            while (np.sum(a + da < 0) > 0) and (test < max_test):
                da *= 0.5
                test += 1
            if test < max_test: 
                a[:] = a + da
            else: 
                print('Warning: parameter a not improved')
        b = np.sum(n*a)/sx if np.sum(a == 0) == 0 else 1
        c = np.max(np.abs(da))
        k += 1
    if (k == max_iteration) and (c > tolerance):
        # print('Warning: bad convergence (b = {})'.format(b))
        a, b = a/b, 1
    if np.sum(a < 0) > 0:
        print('WARNING: a < 0')
    if b < 0: 
        print('WARNING: b < 0')
    if np.all(a == 0): 
        print('WARNING: a == 0')
    # if k > 20 and np.max(a/b) > 2: print(k, np.max(a/b))
    return a, b, k

def infer_proteins(x: np.ndarray, a: np.ndarray) -> np.ndarray:
    """
    Estimate y directly from data.
    """
    nb_cells, nb_genes = x.shape
    y = np.ones((nb_cells, nb_genes))
    z = np.ones((2, nb_genes))
    z[0] = a[0]/a[1]
    z[z<1e-5] = 1e-5
    az = a[1]*z
    for k in range(nb_cells):
        v = az*np.log(a[2]/(a[2]+1)) + gammaln(az+x[k]) - gammaln(az)
        for i in range(1, nb_genes):
            y[k,i] = z[np.argmax(v[:,i]),i]
    # Stimulus off at t <= 0
    y[x[:, 0]<=0, 0] = 0
    return y

def _create_infer_network(objective, grad_theta):
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
    
    return infer_network

infer_network = _create_infer_network(objective, grad_theta)
_infer_network_jit = None

class Hartree(Inference):
    def __init__(self, 
                 penalization_strength: float = 1.0, 
                 tolerance: float = 1e-5, 
                 max_iteration: int = 100, 
                 verbose: bool = False,
                 use_numba: bool = True):
        self.penalization_strength: float = penalization_strength
        self.tolerance: float = tolerance
        self.max_iteration: int = max_iteration
        self.is_verbose: bool = verbose
        # Smoothing threshold
        self.smoothing_threshold: float = 0.1
        self._use_numba, self._infer_network = False, infer_network
        self.use_numba: bool = use_numba

    @property
    def use_numba(self) -> bool:
        return self._use_numba
    
    @use_numba.setter
    def use_numba(self, use_numba: bool) -> None:
        global _infer_network_jit

        if self._use_numba != use_numba:
            if use_numba:
                if _infer_network_jit is None:
                    from numba import njit
                    p1_jit = njit()(p1)
                    grad_p1_jit = njit()(grad_p1)
                    penalization_jit = njit()(_create_penalization(p1_jit))
                    grad_penalization_jit = njit()(
                        _create_grad_penalization(p1_jit, grad_p1_jit))
                    objective_jit = _create_objective(penalization_jit)
                    grad_theta_jit= _create_grad_theta(grad_penalization_jit)
                    _infer_network_jit = _create_infer_network(objective_jit,
                                                               grad_theta_jit)
                self._infer_network = _infer_network_jit
            else:
                self._infer_network = infer_network
            
            self._use_numba = use_numba
            


    def run(self, data: np.ndarray) -> Inference.Result:
        """
        Infers the network model from the data.
        """
        x = data
        # Time points
        times = np.sort(list(set(x[:,0])))
        nb_genes = x.shape[1]

        # Kinetic parameters
        a = self._get_kinetics(data)
        # Concentration parameter
        c = 100 * np.ones(nb_genes)
        # Get protein levels
        y = infer_proteins(x, a)
        # Inference procedure
        theta, nb_iterations =  self._infer_network(x, y, a, c,
                                                    self.penalization_strength,
                                                    self.tolerance,
                                                    self.smoothing_threshold)
        if self.is_verbose: 
            print(f'Fitted theta in {nb_iterations} iterations')
        # Build the results
        basal_time = {time: np.zeros(nb_genes) for time in times}
        inter_time = {time: np.zeros((nb_genes, nb_genes)) for time in times}
        for t, time in enumerate(times):
            basal_time[time] = theta[t][:, 0]
            inter_time[time][:, 1:] = theta[t][:, 1:]

        p = NetworkParameter(nb_genes - 1)
        p.burst_frequency_min, p.burst_frequency_max, p.burst_size_inv = a
        scale = p.scale()
        p.creation_rna = p.degradation_rna * scale
        p.creation_protein = p.degradation_protein * scale
        p.basal = basal_time[times[-1]]
        p.interaction = inter_time[times[-1]]

        return Inference.Result(
            parameter=p,
            basal_time=basal_time,
            interaction_time=inter_time,
            y=y
        )

    def _get_kinetics(self, data: np.ndarray) -> np.ndarray:
        """
        Compute the basal parameters of all genes.
        """
        times = data[:, 0]
        # nb_genes = data[0].size
        nb_genes = data.shape[1]
        # Kinetic values for each gene
        a = np.empty((3, nb_genes))
        a[:, 0] = 1.0
        for g in range(1, nb_genes):
            if self.is_verbose:
                print(f'Calibrating gene {g}...')
            a_g, b_g, k = infer_kinetics(x=data[:, g], 
                                         times=times, 
                                         tolerance=self.tolerance, 
                                         max_iteration=self.max_iteration)
            if self.is_verbose:
                print(f'Estimation done in {k} iterations')
            a[0, g] = np.min(a_g)
            a[1, g] = np.max(a_g)
            a[2, g] = b_g
        return a
    
    def binarize(self, data: np.ndarray) -> np.ndarray:
        """
        Return a binarized version of the data using gene-specific thresholds
        derived from the data-calibrated mechanistic model.
        """
        # Get binarized values (gene-specific thresholds)
        y = infer_proteins(data, self._get_kinetics(data))[:, 1:]
        y = np.floor(y).astype(data.dtype)
        return np.hstack((data[:, 0, np.newaxis], y))
