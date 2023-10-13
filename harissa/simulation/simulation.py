from abc import ABC, abstractmethod
from dataclasses import dataclass
from numpy import ndarray

class Simulation(ABC):
    @dataclass
    class Result:
        time_points: ndarray
        rna: ndarray
        protein: ndarray

    @abstractmethod
    def run(self, 
            time_points: ndarray, 
            burst_frequency_min: ndarray, 
            burst_frequency_max: ndarray, 
            burst_size: ndarray, 
            degradation_rna: ndarray, 
            degradation_protein: ndarray,
            basal: ndarray, 
            interaction: ndarray) -> Result:
        raise NotImplementedError