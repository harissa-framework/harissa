import numpy as np
from scipy.special import expit

def kon(p: np.ndarray,
        basal: np.ndarray,
        inter: np.ndarray,
        k0: np.ndarray,
        k1: np.ndarray) -> np.ndarray:
    """
    Interaction function kon (off->on rate), given protein levels p.
    """
    sigma = expit(basal + p @ inter)
    k_on = (1-sigma)*k0 + sigma*k1
    k_on[0] = 0 # Ignore stimulus
    return k_on
