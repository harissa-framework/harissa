import numpy as np

def estimate_gamma(x):
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
    
def estimate_gamma_poisson(x):
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
    a, b = estimate_gamma_poisson(x)
    if not (a > 0 and b > 0):
        print(('Warning: you should check whether x is not '
            'almost zero (sum(x) = {}).').format(np.sum(x)))
        a, b = np.abs(a), np.abs(b)
    return (a + x)/(b + 1)
