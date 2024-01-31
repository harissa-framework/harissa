"""
Data binarization from the dynamical model
"""
from harissa.inference import Hartree
from harissa.dataset import Dataset

def binarize(data: Dataset) -> Dataset:
    """
    Return a binarized version of the data using gene-specific thresholds
    derived from the data-calibrated mechanistic model. 
    (using the hartree method)
    """
    return Hartree().binarize(data)
