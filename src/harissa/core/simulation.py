from __future__ import annotations
import numpy as np
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

from harissa.core.parameter import NetworkParameter
from harissa.utils.npz_io import load_dir, load_npz, save_dir, save_npz

class Simulation(ABC):
    """
    Abstract class for simulations.
    """
    @dataclass
    class Result:
        """
        Simulation result
        """
        param_names: ClassVar[dict[str, tuple[bool, np.dtype]]] = {
            'time_points': (True, np.float_),
            'rna_levels': (True, np.float_),
            'protein_levels' :(True, np.float_)
        }
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

        @classmethod
        def load_txt(cls, path: str | Path) -> Simulation.Result:
            return cls(**load_dir(path, cls.param_names))
        
        @classmethod
        def load(cls, path: str | Path) -> Simulation.Result:
            return cls(**load_npz(path, cls.param_names))
            
        # Add a "save" methods
        def save_txt(self, path: str | Path) -> Path:
            return save_dir(path, asdict(self))    

        def save(self, path: str | Path) -> Path:
            return save_npz(path, asdict(self))
        
        def __add__(self, result: Simulation.Result):
            if isinstance(result, Simulation.Result):
                if self.time_points[-1] < result.time_points[0]:
                    return Simulation.Result(
                        np.concatenate((self.time_points, result.time_points)),
                        np.concatenate((self.rna_levels, result.rna_levels)),
                        np.concatenate((self.protein_levels, result.protein_levels))
                    )
                else:
                    raise ValueError(('Time points of first element must be '
                                      'lower than that of second element'))
            else:
                raise NotImplementedError

    @abstractmethod
    def run(self, 
            time_points: np.ndarray,
            initial_state: np.ndarray,
            parameter: NetworkParameter) -> Result:
        """Note: here time points must start from 0 (Markov model)."""
        raise NotImplementedError(
            f'{self.__class__.__name__} must only '
             'implement this function (run) and not use it.')
