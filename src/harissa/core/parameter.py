"""
Main class for network parameters
"""
from __future__ import annotations
from typing import Union, Dict
import numpy as np
from pathlib import Path
import json
import re

from harissa.utils.npz_io import (
    ParamInfos,
    load_dir,
    load_npz,
    save_dir,
    save_npz,
    check_names
)

# Default parameter values
default_degradation_rna = np.log(2.0) / 9.0
default_degradation_protein = np.log(2.0) / 46.0
default_burst_frequency_min = 0.0 * default_degradation_rna
default_burst_frequency_max = 2.0 * default_degradation_rna
default_burst_size_inv = 0.02

SPARSE_INTERACTION_KEY_PATTERN = re.compile(
    r'^\s*(\d+|stimulus)\s*->\s*(\d+)\s*$'
)

def serialize_interaction(inter: np.ndarray) -> Dict:
    sparse_interaction = {}
    with np.nditer(inter, flags=['multi_index']) as it:
        for x in it:
            i, j = it.multi_index
            if x != 0.0:
                if i == 0:
                    i = 'stimulus'
                sparse_interaction[f'{i} -> {j}'] = x.item()

    return sparse_interaction

# Main class
class NetworkParameter:
    """
    Parameters of a network model.

    Bursting parameters:
        burst_frequency_min # k[0]
        burst_frequency_max # k[1]
        burst_size_inv      # b

    Creation rates:
        creation_rna        # s[0]
        creation_protein    # s[1]

    Degradation rates:
        degradation_rna     # d[0]
        degradation_protein # d[1]

    Network parameters:
        basal               # beta
        interaction         # theta
    """

    param_names : dict = {
        'burst_frequency_min': ParamInfos(True, np.float64, 1),
        'burst_frequency_max': ParamInfos(True, np.float64, 1),
        'burst_size_inv': ParamInfos(True, np.float64, 1),
        'creation_rna': ParamInfos(True, np.float64, 1),
        'creation_protein': ParamInfos(True, np.float64, 1),
        'degradation_rna': ParamInfos(True, np.float64, 1),
        'degradation_protein': ParamInfos(True, np.float64, 1),
        'basal': ParamInfos(True, np.float64, 1),
        'interaction': ParamInfos(True, np.float64, 2),
        'gene_names': ParamInfos(False, np.str_, 1),
        'layout': ParamInfos(False, np.float64, 2)
    }

    def __init__(self, n_genes, gene_names=None, layout=None):
        # Number of genes
        self._n_genes = _check_n_genes(n_genes)
        # Genes plus stimulus
        G = self.n_genes_stim
        # Initialize parameters
        self._burst = _masked_zeros((3, G))
        self._creation = _masked_zeros((2, G))
        self._degradation = _masked_zeros((2, G))
        self._basal = _masked_zeros(G)
        self._interaction = _masked_zeros((G, G))
        # Default bursting parameters
        self._burst[0] = default_burst_frequency_min
        self._burst[1] = default_burst_frequency_max
        self._burst[2] = default_burst_size_inv
        # Default degradation rates
        self._degradation[0] = default_degradation_rna
        self._degradation[1] = default_degradation_protein
        # Default creation rates
        self._creation[0] = self._degradation[0] * self.rna_scale()
        self._creation[1] = self._degradation[1] * self.protein_scale()

        self._gene_names = _check_gene_names(gene_names, G)
        self._layout = _check_layout(layout, G)

    def __eq__(self, other):
        if isinstance(other, NetworkParameter):
            test = [other._n_genes == self._n_genes]
            for k in self._array_names():
                test.append(bool(
                    np.all(getattr(other, k) == getattr(self, k))))
            return all(test)

        raise NotImplementedError

    @classmethod
    def load_txt(cls, path: Union[str, Path]) -> NetworkParameter:
        data = load_dir(path, cls.param_names)
        gene_names = data.pop('gene_names', None)
        layout = data.pop('layout', None)
        network_param = cls(data['basal'].size - 1, gene_names, layout)

        for key, value in data.items():
            getattr(network_param, key)[:] = value[:]

        return network_param

    @classmethod
    def load(cls, path: Union[str, Path]) -> NetworkParameter:
        data = load_npz(path, cls.param_names)
        gene_names = data.pop('gene_names', None)
        layout = data.pop('layout', None)
        network_param = cls(data['basal'].size - 1, gene_names, layout)

        for key, value in data.items():
            getattr(network_param, key)[:] = value[:]

        return network_param

    @classmethod
    def load_json(cls, path : Union[str, Path]) -> NetworkParameter:
        path = Path(path).with_suffix('.json')
        with open(path, 'r') as fp:
            data = json.load(fp)
        n_genes = len(data['basal'])
        check_names(data.keys(), cls.param_names)
        data_optional = dict(
            map(
                lambda i: (i[0], data.pop(i[0], None)),
                filter(lambda i: not i[1].required, cls.param_names.items())
            )
        )
        kwargs = {
            k:(
                np.empty(
                    ((n_genes + 1,)
                    if cls.param_names[k].ndim == 1 else
                    (n_genes + 1, 2)),
                    dtype=cls.param_names[k].dtype
                    if cls.param_names[k].dtype != np.str_ else
                    f'U{len(max(v.values(), key=len))}'
                )
                if v is not None else v
            )
            for k, v in data_optional.items()
        }

        for key, value in data_optional.items():
            if value is not None:
                if len(value) == n_genes + 1:
                    for k, v in value.items():
                        i = 0 if k == 'stimulus' else int(k)
                        if isinstance(v, list):
                            kwargs[key][i] = np.array(v)
                        else:
                            kwargs[key][i] = v
                else:
                    raise RuntimeError(f'{key} must have {n_genes+1} items')


        spare_interaction = data.pop('interaction')

        network_param = cls(n_genes, **kwargs)

        for key, value in data.items():
            if len(value) == n_genes:
                for k, v in value.items():
                    getattr(network_param, key)[int(k)] = v
            else:
                raise RuntimeError(f'{key} must have {n_genes} items')

        marked_indices = []
        for key, value in spare_interaction.items():
            res = SPARSE_INTERACTION_KEY_PATTERN.fullmatch(key)
            if res is not None:
                i = int(res.group(1)) if res.group(1) != 'stimulus' else 0
                j = int(res.group(2))
                if (i, j) not in marked_indices:
                    network_param.interaction[i, j] = value
                    marked_indices.append((i, j))
                else:
                    raise RuntimeError(
                        f'{res.group(1)} -> {res.group(2)} must be unique.'
                    )
            else:
                raise RuntimeError('key format invalid. TODO find better msg')

        return network_param

    def as_dict(self):
        param_dict = {}
        for attr_name, attr_info in self.param_names.items():
            attr_array = getattr(self, attr_name)
            if attr_info.required or attr_array is not None:
                param_dict[attr_name] = attr_array

        return param_dict

    def as_serialized_dict(self):
        return {
            k:({
                (i if i != 0 else 'stimulus'):x.tolist()
                for i, x in enumerate(v) if x is not np.ma.masked
            } if k != 'interaction' else serialize_interaction(
                self.interaction.filled(0.0)
            ))
            for k,v in self.as_dict().items()
        }

    def save_txt(self, path: Union[str, Path]) -> Path:
        return save_dir(path, self.as_dict())

    def save(self, path: Union[str, Path]) -> Path:
        return save_npz(path, self.as_dict())

    def save_json(self, path: Union[str, Path]) -> Path:
        path = Path(path).with_suffix('.json')
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as fp:
            json.dump(self.as_serialized_dict(), fp, indent=4)

        return path


    # Network size properties
    # =======================

    @property
    def n_genes(self):
        """Number of genes in the network model, without stimulus."""
        return self._n_genes

    @property
    def n_genes_stim(self):
        """Number of genes in the network model, including stimulus."""
        return self._n_genes + 1

    @property
    def gene_names(self):
        return self._gene_names

    @gene_names.setter
    def gene_names(self, gene_names):
        self._gene_names = _check_gene_names(gene_names, self.n_genes_stim)

    @property
    def layout(self):
        return self._layout

    @layout.setter
    def layout(self, layout):
        self._layout = _check_layout(layout, self.n_genes_stim)

    # Parameter properties
    # ====================

    @property
    def burst_frequency_min(self):
        """Minimal bursting frequency for each gene (low expression)."""
        return self._burst[0]

    @property
    def burst_frequency_max(self):
        """Maximal bursting frequency for each gene (high expression)."""
        return self._burst[1]

    @property
    def burst_size_inv(self):
        """Inverse of average burst size for each gene."""
        return self._burst[2]

    @property
    def creation_rna(self):
        """mRNA creation rates. Note that in the transcriptional
        bursting regime, s[0] is not identifiable since it aggregates with
        koff (inverse of average ON time) into parameter b = s[0]/koff."""
        return self._creation[0]

    @property
    def creation_protein(self):
        """Protein creation rates."""
        return self._creation[1]

    @property
    def degradation_rna(self):
        """mRNA degradation rates."""
        return self._degradation[0]

    @property
    def degradation_protein(self):
        """Protein degradation rates."""
        return self._degradation[1]

    @property
    def basal(self):
        """Basal expression level for each gene."""
        return self._basal

    @property
    def interaction(self):
        """Interactions between genes."""
        return self._interaction

    # Aliases (mathematical notation)
    # ===============================

    @property
    def k(self):
        """Bursting frequency bounds for each gene."""
        return self._burst[:2]

    @property
    def b(self):
        """Inverse of average burst size for each gene."""
        return self._burst[2]

    @property
    def s(self):
        """mRNA and protein creation rates. Note that in the transcriptional
        bursting regime, s[0] is not identifiable since it aggregates with
        koff (inverse of average ON time) into parameter b = s[0]/koff."""
        return self._creation

    @property
    def d(self):
        """mRNA and protein degradation rates."""
        return self._degradation

    @property
    def a(self):
        """Bursting kinetics (not normalized)."""
        return self._burst

    @property
    def beta(self):
        """Basal expression level for each gene."""
        return self._basal

    @property
    def theta(self):
        """Interactions between genes."""
        return self._interaction

    # @property
    # def k_normalized(self):
    #     """Bursting kinetics (not normalized)."""
    #     return self._burst

    # Shortcut methods
    # ================

    def c(self):
        return self.b * self.d[0] / self.s[1]

    def rna_scale(self):
        return self.b / self.k[1]
    def protein_scale(self):
        return self.d[0] * self.b / self.k[1]

    # Methods
    # =======

    def _array_names(self):
        """Attribute names of the underlying array parameters."""
        ma = np.ma.MaskedArray
        return [k for k, v in vars(self).items() if isinstance(v, ma)]

    def copy(self, shallow=False):
        """Return a copy of the network parameters.
        If shallow is False (default), underlying arrays are fully copied.
        If shallow is True, underlying arrays are only referenced."""
        new_param = NetworkParameter(self._n_genes)
        for k in self._array_names():
            if shallow:
                setattr(new_param, k, getattr(self, k))
            else:
                getattr(new_param, k)[:] = getattr(self, k)
        return new_param


# Utility functions
# =================

def _check_n_genes(arg):
    if isinstance(arg, int):
        if arg > 0:
            return arg
        else:
            raise ValueError('n_genes must be strictly positive.')
    raise TypeError((f'n_genes of type {type(arg).__name__} '
                     'must be an integer.'))

def _check_gene_names(gene_names, n_genes_stim):
    if gene_names is not None:
        if (not isinstance(gene_names, np.ndarray)
            or gene_names.ndim != 1
            or gene_names.dtype.type is not np.str_):
            raise TypeError('gene_names must be a str 1D ndarray.')

        if gene_names.size != n_genes_stim:
            raise TypeError(
                'gene_names size must be equal to the number of genes'
                ' (including the stimulus)'
                f'({gene_names.size} != {n_genes_stim})'
            )

    return gene_names

def _check_layout(layout, n_genes_stim):
    if layout is not None:
        if (not isinstance(layout, np.ndarray)
            or layout.ndim != 2
            or layout.dtype != np.float64):
            raise TypeError('layout must be a float 2D ndarray.')

        if layout.shape[0] != n_genes_stim:
            raise TypeError(
                'layout number of row must be equal to the number of genes'
                ' (including the stimulus)'
                f'({layout.size} != {n_genes_stim})'
            )

        if layout.shape[1] != 2:
            raise TypeError(
                'layout number of colum must be equal 2'
                f'({layout.size} != 2)'
            )

    return layout

def _masked_zeros(shape):
    """Array of zeros with given shape and hard-masked first column.
    Note that np.zeros is used instead of np.empty in order to avoid
    a runtime warning problem with numpy.ma operations."""
    mask = np.zeros(shape, dtype=bool)
    mask[..., 0] = True # Handle both 1D and 2D arrays
    return np.ma.array(np.zeros(shape), mask=mask, hard_mask=True)
