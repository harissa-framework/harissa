"""
Data binarization from the dynamical model
"""
from harissa.inference.hartree import Hartree
from harissa.core.dataset import Dataset

def binarize(data: Dataset) -> Dataset:
    """
    Return a binarized version of the data using gene-specific thresholds
    derived from the data-calibrated mechanistic model. 
    """
    return Hartree().binarize(data)
