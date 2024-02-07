import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod
from harissa.core.parameter import NetworkParameter

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

        @property
        def stimulus_levels(self):
            return self.protein_levels[:, 0]

        @property
        def final_state(self):
            # state: row 0 <-> rna, row 1 <-> protein
            state = np.zeros((2, self.rna_levels.shape[1]))
            state[0] = self.rna_levels[-1]
            state[1] = self.protein_levels[-1]
            return state

        # # Add a "save" methods
        # def save_txt(self):
        #     pass

        # def save(self):
        #     pass

    @abstractmethod
    def run(self, 
            initial_state: np.ndarray,
            time_points: np.ndarray,
            parameter: NetworkParameter) -> Result:
        raise NotImplementedError(
            f'{self.__class__.__name__} must only '
             'implement this function (run) and not use it.')
