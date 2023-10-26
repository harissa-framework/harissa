import numpy as np

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