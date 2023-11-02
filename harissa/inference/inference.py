from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np

class Inference(ABC):
    @dataclass
    class Result:
        """
        Result of inference
        """
        # Kinetic parameters
        burst_frequency_min : np.ndarray | None = None
        burst_frequency_max : np.ndarray | None = None
        burst_size          : np.ndarray | None = None
        creation_rna        : np.ndarray | None = None
        creation_protein    : np.ndarray | None = None
        degradation_rna     : np.ndarray | None = None
        degradation_protein : np.ndarray | None = None
        
        # Network parameters
        basal       : np.ndarray | None = None
        interaction : np.ndarray | None = None

    @abstractmethod
    def run(self, data: np.ndarray) -> Result:              
        raise NotImplementedError(
            f'{self.__class__.__name__} must only ' 
             'implement this function (run) and not use it.')