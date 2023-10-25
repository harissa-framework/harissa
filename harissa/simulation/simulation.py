from abc import ABC, abstractmethod
from dataclasses import dataclass
from numpy import ndarray

class Simulation(ABC):
    @dataclass
    class Result:
        """
        Result of Simulation
        """
        time_points: ndarray
        rna_levels: ndarray
        protein_levels: ndarray

    @abstractmethod
    def run(self,
            initial_state: ndarray, 
            time_points: ndarray, 
            burst_frequency_min: ndarray, 
            burst_frequency_max: ndarray, 
            burst_size: ndarray, 
            degradation_rna: ndarray, 
            degradation_protein: ndarray,
            basal: ndarray, 
            interaction: ndarray) -> Result:
        raise NotImplementedError