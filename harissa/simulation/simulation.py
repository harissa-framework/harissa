from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np

class Simulation(ABC):
    @dataclass
    class Parameter:
        """
        Parameters of simulation
        """
        initial_state: np.ndarray = field(init=False)
        time_points: np.ndarray = field(init=False)
        burst_frequency_min: np.ndarray
        burst_frequency_max: np.ndarray
        burst_size: np.ndarray
        degradation_rna: np.ndarray
        degradation_protein: np.ndarray
        basal: np.ndarray
        interaction: np.ndarray

    @dataclass
    class Result:
        """
        Result of simulation
        """
        time_points: np.ndarray
        rna_levels: np.ndarray
        protein_levels: np.ndarray

    @abstractmethod
    def run(self, parameter: Parameter) -> Result:
        raise NotImplementedError(
            f'{self.__class__.__name__} must only ' 
             'implement this function (run) and not use it.')