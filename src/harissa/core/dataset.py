from __future__ import annotations
from typing import Dict, Union, Optional, Literal
import numpy as np
from scipy.sparse import issparse
from pandas import DataFrame
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

_anndata_msg_error = 'Install the package anndata to use this function.'

@dataclass(frozen=True, init=False)
class Dataset:
    param_names: ClassVar[Dict[str, ParamInfos]] = {
        'time_points': ParamInfos(True, np.float64, 1),
        'count_matrix': ParamInfos(True, np.uint, 2),
        'gene_names': ParamInfos(False, np.str_, 1)
    }
    time_points: np.ndarray
    count_matrix: np.ndarray
    gene_names: Optional[np.ndarray]

    def __init__(self,
        time_points: np.ndarray,
        count_matrix: np.ndarray,
        gene_names=None
    ) -> None:
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

    def save_txt(self, path: Union[str, Path]) -> Path:
        return save_dir(path, self.as_dict())

    def save(self, path: Union[str, Path]) -> Path:
        return save_npz(path, self.as_dict())

    def as_dict(self) -> Dict[str, np.ndarray]:
        return asdict(
            self,
            dict_factory=lambda x: {k:v for (k, v) in x if v is not None}
        )

    @classmethod
    def from_annData(cls, adata) -> Dataset:
        try:
            from anndata import AnnData

            if not isinstance(adata, AnnData):
                raise RuntimeError('adata must be an AnnData object.')

            if isinstance(adata.X, DataFrame):
                count_matrix = adata.X.to_numpy()
            elif issparse(adata.X):
                count_matrix = adata.X.toarray()
            else:
                count_matrix = adata.X

            time_points = adata.obs.get('time_points', None)
            if time_points is None:
                raise RuntimeError(
                    'adata must have a time_points field in its obs.'
                )

            return cls(
                np.array(time_points, dtype=np.float64),
                count_matrix.astype(np.uint),
                np.array(adata.var_names, dtype=np.str_)
            )
        except ImportError:
            raise RuntimeError(_anndata_msg_error)

    @classmethod
    def load_h5ad(cls, path: Union[str, Path]) -> Dataset:
        try:
            from anndata import read_h5ad
            return cls.from_annData(read_h5ad(path))
        except ImportError:
            raise RuntimeError(_anndata_msg_error)

    def as_annData(self):
        try:
            from anndata import AnnData
            adata = AnnData(
                self.count_matrix,
                {'time_points': self.time_points}
            )
            adata.obs_names = np.array(
                [f'Cell_{i+1}' for i in range(adata.n_obs)]
            )
            if self.gene_names is not None:
                adata.var_names = self.gene_names
            else:
                adata.var_names = np.array(
                    [f'Gene_{i}' for i in range(adata.n_vars)]
                )

            return adata
        except ImportError:
            raise RuntimeError(_anndata_msg_error)

    def save_h5ad(self,
        path: Union[str, Path],
        compression: Optional[Literal['gzip', 'lzf']] = None,
        compression_opts = None
    ) -> Path:
        path = Path(path)
        adata = self.as_annData()
        adata.write_h5ad(path, compression, compression_opts)
        return path
