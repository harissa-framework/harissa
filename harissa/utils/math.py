import numpy as np

## --------------------------
## ------- Inference --------
## --------------------------

def estim_gamma(x):
    """
    Estimate the parameters of a gamma distribution using
    the method of moments. The output is (a,b) for the distribution
    f(x) = x**(a-1)*exp(-b*x)/(gamma(a)/b**a).
    """
    m = np.mean(x)
    v = np.var(x)
    if v == 0: 
        return 0, 1
    else: 
        return m*m/v, m/v
    
def estim_gamma_poisson(x):
    """
    Estimate parameters a and b of the Gamma-Poisson(a,b) distribution,
    a.k.a. negative binomial distribution, using the method of moments.
    """
    m1 = np.mean(x)
    m2 = np.mean(x*(x-1))
    if m1 == 0: 
        return 0, 1
    r = m2 - m1**2
    if r > 0: 
        b = m1/r
    else:
        v = np.var(x)
        if v == 0: 
            return 0, 1
        b = m1/v
    a = m1 * b
    return a, b

def transform(x):
    """
    Replace x by the conditional expectation given x of the underlying
    Gamma distribution, within the Gamma-Poisson model inferred from x.
    NB: this simply corresponds to a linear transformation with offset.
    """
    a, b = estim_gamma_poisson(x)
    if not (a > 0 and b > 0):
        print(('Warning: you should check whether x is not '
            'almost zero (sum(x) = {}).').format(np.sum(x)))
        a, b = np.abs(a), np.abs(b)
    return (a + x)/(b + 1)


## --------------------------
## ------ Simulation --------
## --------------------------

def kon(p: np.ndarray, 
        basal: np.ndarray, 
        inter: np.ndarray, 
        k0: np.ndarray, 
        k1: np.ndarray) -> np.ndarray:
    """
    Interaction function kon (off->on rate), given protein levels p.
    """
    phi = np.exp(basal + p @ inter)
    k_on = (k0 + k1*phi)/(1 + phi)
    k_on[0] = 0 # Ignore stimulus
    return k_on

def kon_bound(state: np.ndarray, 
              basal: np.ndarray, 
              inter: np.ndarray, 
              d0: np.ndarray, 
              d1: np.ndarray, 
              s1: np.ndarray, 
              k0: np.ndarray, 
              k1: np.ndarray) -> np.ndarray:
    """
    Compute the current kon upper bound.
    """
    m, p = state
    # Explicit upper bound for p
    time = np.log(d0/d1)/(d0-d1) # vector of critical times
    p_max = p + (s1/(d0-d1))*m*(np.exp(-time*d1) - np.exp(-time*d0))
    p_max[0] = p[0] # Discard stimulus
    # Explicit upper bound for Kon
    phi = np.exp(basal + p_max @ ((inter > 0) * inter))
    k_on = (k0 + k1*phi)/(1 + phi) + 1e-10 # Fix precision errors
    k_on[0] = 0 # Ignore stimulus
    return k_on

def flow(time: float,
         state: np.ndarray,
         d0: np.ndarray,
         d1: np.ndarray,
         s1: np.ndarray) -> np.ndarray:
    """
    Deterministic flow for the bursty model.
    """
    m, p = state
    # Explicit solution of the ode generating the flow
    m_new = m*np.exp(-time*d0)
    p_new = ((s1/(d0-d1))*m*(np.exp(-time*d1) - np.exp(-time*d0))
            + p*np.exp(-time*d1))
    m_new[0], p_new[0] = m[0], p[0] # discard stimulus
    return np.vstack((m_new, p_new))