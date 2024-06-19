from __future__ import annotations
from typing import Dict, Union
import numpy as np
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import ClassVar

from harissa.utils.npz_io import (
    ParamInfos,
    load_dir,
    load_npz,
    save_dir,
    save_npz
) 

@dataclass(frozen=True, init=False)
class Dataset:
    param_names: ClassVar[Dict[str, ParamInfos]] = {
        'time_points': ParamInfos(True, np.float64, 1), 
        'count_matrix': ParamInfos(True, np.uint, 2),
        'gene_names': ParamInfos(False, np.str_, 1)
    }
    time_points: np.ndarray
    count_matrix: np.ndarray
    gene_names: np.ndarray

    def __init__(self, 
                 time_points: np.ndarray, 
                 count_matrix: np.ndarray,
                 gene_names=None) -> None:

        if not (time_points.ndim == 1 and 
                time_points.dtype == self.param_names['time_points'].dtype):
            raise TypeError('time_points must be a float 1D ndarray.')

        if not (count_matrix.ndim == 2 and 
                count_matrix.dtype == self.param_names['count_matrix'].dtype):
            raise TypeError('count_matrix must be an uint 2D ndarray.')

        if time_points.shape[0] != count_matrix.shape[0]:
            raise TypeError(
                'time_points must have the same number of elements' 
                ' than the rows of count_matrix.' 
                f'({time_points.shape[0]} != {count_matrix.shape[0]})'
            )

        if count_matrix.shape[1] <= 1:
            raise TypeError('count_matrix must have at least 2 columns.')
        
        if gene_names is not None:
            if (gene_names.ndim != 1 or
                gene_names.dtype.type is not np.str_):
                raise TypeError('gene_names must be a str 1D ndarray.')
             
            if gene_names.shape[0] != count_matrix.shape[1]:
                raise TypeError(
                    'genes_names must have the same number of elements' 
                    ' than the columns of count_matrix.' 
                    f'({gene_names.shape[0]} != {count_matrix.shape[1]})'
                )

        object.__setattr__(self, 'time_points', time_points)
        object.__setattr__(self, 'count_matrix', count_matrix)
        object.__setattr__(self, 'gene_names', gene_names)

    # Add load method
    @classmethod
    def load_txt(cls, path: Union[str, Path]) -> Dataset:
        path = Path(path) # convert it to Path (needed for str)
        if path.suffix == '.txt':
            if not path.exists():
                raise RuntimeError(f"{path} doesn't exist.")
            # Backward compatibility, dataset inside a txt file.
            # It assumes that the 1rst column is the time points (arr_list[0]) 
            # and the rest is the count matrix (arr_list[1])
            arr = np.loadtxt(path)
            data_list = [
                arr[:, 0].copy(),     # time points
                arr.astype(np.uint),  # count matrix
                None                  # gene names
            ]

            # Set stimuli instead of time_points
            data_list[1][:, 0] = data_list[0] != 0.0
            data = {}
            for i, name in enumerate(cls.param_names):
                data[name] = data_list[i]
        else:
            data = load_dir(path, cls.param_names)

        return cls(**data)

    @classmethod
    def load(cls, path: Union[str, Path]) -> Dataset:
        return cls(**load_npz(path, cls.param_names))
    
    def as_dict(self) -> Dict[str, np.ndarray]:
        return asdict(
            self, 
            dict_factory=lambda x: {k:v for (k, v) in x if v is not None}
        )
    
    # Add a "save" methods
    def save_txt(self, path: Union[str, Path]) -> Path:
        return save_dir(path, self.as_dict())

    def save(self, path: Union[str, Path]) -> Path:
        return save_npz(path, self.as_dict())

