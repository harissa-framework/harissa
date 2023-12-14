import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod
from harissa.parameter import NetworkParameter

class Simulation(ABC):
    """
    Abstract class for simulations.
    """
    @dataclass
    class Result:
        """
        Simulation result
        """
        time_points: np.ndarray
        rna_levels: np.ndarray
        protein_levels: np.ndarray

    @abstractmethod
    def run(self, 
            initial_state: np.ndarray,
            time_points: np.ndarray,
            parameter: NetworkParameter) -> Result:
        raise NotImplementedError(
            f'{self.__class__.__name__} must only '
             'implement this function (run) and not use it.')
