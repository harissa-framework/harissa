"""
Core functions for network inference using likelihood maximization
"""
from .inference import Inference
from ..utils.math import estim_gamma_poisson
import numpy as np
from scipy.special import psi, polygamma, expit, gammaln
from scipy.optimize import minimize

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

def infer_proteins(x, a):
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

class Hartree(Inference):    
    def __init__(self, 
                 penalization_strength=1, 
                 tolerance=1e-5, 
                 max_iteration=100, 
                 verbose=False):
        self.penalization_strength = penalization_strength
        self.tolerance = tolerance
        self.max_iteration = max_iteration
        self.is_verbose = verbose
        # Smoothing threshold
        self.smoothing_threshold = 0.1


    def run(self, data: np.ndarray) -> Inference.Result:
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
        theta = self.infer_network(x, y, a, c)
        # Build the results
        # basal = np.zeros(nb_genes)
        # inter = np.zeros((nb_genes, nb_genes))
        basal_time = {time: np.zeros(nb_genes) for time in times}
        inter_time = {time: np.zeros((nb_genes, nb_genes)) for time in times}
        for t, time in enumerate(times):
            basal_time[time][:] = theta[t][:,0]
            inter_time[time][:,1:] = theta[t][:,1:]
        # basal[:] = theta[-1][:,0]
        # inter[:,1:] = theta[-1][:,1:]
        res =  Inference.Result(burst_frequency_min= a[0], 
                                burst_frequency_max= a[1], 
                                burst_size = a[2],
                                basal = theta[-1, :, 0],
                                interaction=theta[-1, :, 1:])
        res.basal_time = basal_time
        res.interaction_time = inter_time
        res.y = y
        return res
        # TODO test this syntax
        # return Inference.Result(burst_frequency_min= a[0], 
        #                         burst_frequency_max= a[1], 
        #                         burst_size = a[2],
        #                         basal = theta[-1, :, 0], 
        #                         basal_time = basal_time,
        #                         interaction=theta[-1, :, 1:],
        #                         interaction_time= inter_time)

    def get_kinetics(self, data: np.ndarray):
        """
        Compute the basal parameters of all genes.
        """
        times = data[:,0]
        nb_genes = data[0].size
        # Kinetic values for each gene
        a = np.ones((3, nb_genes))
        for g in range(1, nb_genes):
            if self.is_verbose: 
                print(f'Calibrating gene {g}...')
            x = data[:,g]
            at, b = self.infer_kinetics(x, times)
            a[0,g] = np.min(at)
            a[1,g] = np.max(at)
            a[2,g] = b
        return a

    def infer_kinetics(self, x, times):
        """
        Infer parameters a[0], ..., a[m-1] and b of a Gamma-Poisson 
        model with time-dependant a and constant b 
        for a given gene at m time points.

        Parameters
        ----------
        x[k] = gene expression in cell k
        times[k] = time point of cell k
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
            a[i], b[i] = estim_gamma_poisson(x[cells])
        b = np.mean(b)
        # Newton-like method
        k, c = 0, 0
        sx = np.sum(x)
        while (k == 0) or (k < self.max_iteration and c > self.tolerance):
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
        if (k == self.max_iteration) and (c > self.tolerance):
            # print('Warning: bad convergence (b = {})'.format(b))
            a, b = a/b, 1
        # if verb: print('Estimation done in {} iterations'.format(k))
        if np.sum(a < 0) > 0:
            print('WARNING: a < 0')
        if b < 0: 
            print('WARNING: b < 0')
        if np.all(a == 0): 
            print('WARNING: a == 0')
        # if k > 20 and np.max(a/b) > 2: print(k, np.max(a/b))
        return a, b
    
    def infer_network(self, x, y, a, c):
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
        if self.tolerance is not None: 
            params['tol'] = self.tolerance
        # Inference routine
        for t, time in enumerate(times):
            res = minimize(objective, theta0.reshape(nb_genes**2),
                    args=(theta0, 
                          x[k==time], 
                          y[k==time], 
                          a, c, d, 
                          self.penalization_strength, 
                          t, 
                          self.smoothing_threshold),
                    jac=grad_theta, **params)
            if not res.success:
                print(f'Warning: maximization failed (time {t})')
            # Update theta0
            theta0 = res.x.reshape((nb_genes, nb_genes))
            # Store theta at time t
            theta[t] = theta0
        if self.is_verbose: 
            print(f'Fitted theta in {res.nit} iterations')
        return theta