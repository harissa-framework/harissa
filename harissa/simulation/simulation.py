from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np

class Simulation(ABC):
    @dataclass
    class Result:
        """
        Result of Simulation
        """
        time_points: np.ndarray
        rna_levels: np.ndarray
        protein_levels: np.ndarray

    @abstractmethod
    def run(self,
            initial_state: np.ndarray, 
            time_points: np.ndarray, 
            burst_frequency_min: np.ndarray, 
            burst_frequency_max: np.ndarray, 
            burst_size: np.ndarray, 
            degradation_rna: np.ndarray, 
            degradation_protein: np.ndarray,
            basal: np.ndarray, 
            interaction: np.ndarray) -> Result:
        raise NotImplementedError(
            f'{self.__class__.__name__} must only ' 
             'implement this function (run) and not use it.')