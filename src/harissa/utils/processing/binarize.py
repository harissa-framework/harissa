"""
Data binarization from the dynamical model
"""
import numpy as np
from harissa.inference.hartree.hartree import Hartree

def binarize(data: np.ndarray) -> np.ndarray:
    """
    Return a binarized version of the data using gene-specific thresholds
    derived from the data-calibrated mechanistic model. 
    (using the hartree method)
    """
    return Hartree().binarize(data)