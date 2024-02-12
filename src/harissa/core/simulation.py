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

    @abstractmethod
    def run(self, 
            initial_state: np.ndarray,
            time_points: np.ndarray,
            parameter: NetworkParameter) -> Result:
        raise NotImplementedError(
            f'{self.__class__.__name__} must only '
             'implement this function (run) and not use it.')