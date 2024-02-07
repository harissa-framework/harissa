import numpy as np
from dataclasses import dataclass

@dataclass(frozen=True, init=False)
class Dataset:
    time_points: np.ndarray
    count_matrix: np.ndarray

    def __init__(self, 
                 time_points: np.ndarray, 
                 count_matrix: np.ndarray,
                 gene_names=None) -> None:

        if not (time_points.ndim == 1 and time_points.dtype == np.float_):
            raise TypeError('time_points must be a float 1D ndarray.')

        if not (count_matrix.ndim == 2 and count_matrix.dtype == np.uint):
            raise TypeError('count_matrix must be an uint 2D ndarray.')

        if time_points.shape[0] != count_matrix.shape[0]:
            raise TypeError(
                'time_points must have the same number of elements' 
                ' than the rows of count_matrix.' 
                f'({time_points.shape[0]} != {count_matrix.shape[0]})'
            )

        if count_matrix.shape[1] <= 1:
            raise TypeError('count_matrix must have at least 2 columns.')

        object.__setattr__(self, 'time_points', time_points)
        object.__setattr__(self, 'count_matrix', count_matrix)
        object.__setattr__(self, 'gene_names', gene_names)

        # # Add a "save" methods
        # def save_txt(self):
        #     pass

        # def save(self):
        #     pass
