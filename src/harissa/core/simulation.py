from __future__ import annotations
from typing import Dict, ClassVar, Union
import numpy as np
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from pathlib import Path

from harissa.core.parameter import NetworkParameter
from harissa.utils.npz_io import (
    ParamInfos, 
    load_dir, 
    load_npz, 
    save_dir, 
    save_npz
)

class Simulation(ABC):
    """
    Abstract class for simulations.
    """
    @dataclass(init=False)
    class Result:
        """
        Simulation result
        """
        param_names: ClassVar[Dict[str, ParamInfos]] = {
            'time_points': ParamInfos(True, np.float64, 1),
            'rna_levels': ParamInfos(True, np.float64, 2),
            'protein_levels' :ParamInfos(True, np.float64, 2)
        }
        time_points: np.ndarray
        rna_levels: np.ndarray
        protein_levels: np.ndarray

        def __init__(self, time_points, rna_levels, protein_levels) -> None:
            if (not isinstance(time_points, np.ndarray) 
                or time_points.ndim != 1
                # or time_points.dtype != np.float64
                ):
                raise TypeError('time_points must be a 1D float np.ndarray')
            
            if (not isinstance(rna_levels, np.ndarray) 
                or rna_levels.ndim != 2
                # or rna_levels.dtype != np.float64
                ):
                raise TypeError('rna_levels must be a 2D float np.ndarray')
            
            if (not isinstance(protein_levels, np.ndarray) 
                or protein_levels.ndim != 2
                # or protein_levels.dtype != np.float64
                ):
                raise TypeError('protein_levels must be a 2D float np.ndarray')
            
            if not np.array_equal(time_points, np.unique(time_points)):
                raise ValueError('time_points must be sorted and unique')
            
            if time_points.shape[0] != rna_levels.shape[0]:
                raise ValueError('The number of time points must' 
                                 'be equal to number of rna levels')
            
            if rna_levels.shape != protein_levels.shape:
                raise ValueError('rna_levels and protein_levels'
                                 'shape must be equal')
            

            self.time_points = time_points
            self.rna_levels = rna_levels
            self.protein_levels = protein_levels

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
        def load_txt(cls, path: Union[str, Path]) -> Simulation.Result:
            return cls(**load_dir(path, cls.param_names))
        
        @classmethod
        def load(cls, path: Union[str, Path]) -> Simulation.Result:
            return cls(**load_npz(path, cls.param_names))
            
        # Add a "save" methods
        def save_txt(self, path: Union[str, Path]) -> Path:
            return save_dir(path, asdict(self))    

        def save(self, path: Union[str, Path]) -> Path:
            return save_npz(path, asdict(self))
        
        def __add__(self, result: Simulation.Result):
            if isinstance(result, Simulation.Result):
                if self.rna_levels.shape == result.rna_levels.shape:
                    if self.time_points[-1] < result.time_points[0]:
                        return Simulation.Result(
                            np.concatenate((
                                self.time_points,
                                result.time_points
                            )),
                            np.concatenate((
                                self.rna_levels,
                                result.rna_levels
                            )),
                            np.concatenate((
                                self.protein_levels,
                                result.protein_levels
                            ))
                        )
                    else:
                        raise ValueError(
                            ('Time points of first element must be '
                             'lower than that of second element.')
                        )
                else:
                    raise ValueError('The shapes of rna levels are not equal.')
            else:
                raise NotImplementedError('The right operand must be a '
                                          'Simulation.Result object.')

    @abstractmethod
    def run(self, 
            time_points: np.ndarray,
            initial_state: np.ndarray,
            parameter: NetworkParameter) -> Result:
        """Note: here time points must start from 0 (Markov model)."""
        raise NotImplementedError(
            f'{self.__class__.__name__} must only '
             'implement this function (run) and not use it.')
