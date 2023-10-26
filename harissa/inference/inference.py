from abc import ABC, abstractmethod
from dataclasses import dataclass
from numpy import ndarray

class Inference(ABC):
    @dataclass
    class Result:
        """
        Result of inference
        """
        # Kinetic parameters
        burst_frequency_min : ndarray | None = None
        burst_frequency_max : ndarray | None = None
        burst_size          : ndarray | None = None
        creation_rna        : ndarray | None = None
        creation_protein    : ndarray | None = None
        degradation_rna     : ndarray | None = None
        degradation_protein : ndarray | None = None
        
        # Network parameters
        basal       : ndarray | None = None
        interaction : ndarray | None = None

    @abstractmethod
    def run(self, data: ndarray) -> Result:
        raise NotImplementedError(
            f'{self.__class__.__name__} must only ' 
             'implement this function (run) and not use it.')