"""
Main class for network parameters
"""
import numpy as np

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
    def __init__(self, n_genes):
        # Number of genes
        self._n_genes = _check_n_genes(n_genes)
        # Mask for ignoring stimulus
        G = self.n_genes_stim
        mask = np.ma.make_mask_none((G,G))
        mask[:, 0] = True
        # Initialize parameters
        self._burst_frequency = _masked_zeros((2,G), mask[:2])
        self._burst_size_inv = _masked_zeros(G, mask[0])
        self._creation = _masked_zeros((2,G), mask[:2])
        self._degradation = _masked_zeros((2,G), mask[:2])
        self._basal = _masked_zeros(G, mask[0])
        self._interaction = _masked_zeros((G,G), mask)
        # Default bursting parameters
        self._burst_frequency[0] = 0.0
        self._burst_frequency[1] = 2.0
        self._burst_size_inv[:] = 0.02
        # Default degradation rates
        self._degradation[0] = np.log(2.0) / 9.0
        self._degradation[1] = np.log(2.0) / 46.0
        # Default creation rates
        scale = self.scale
        self._creation[0] = self._degradation[0] * scale
        self._creation[1] = self._degradation[1] * scale

    def __eq__(self, other):
        if isinstance(other, NetworkParameter):
            test = [other._n_genes == self._n_genes]
            for k in self._array_names():
                test.append(np.all(getattr(other, k) == getattr(self, k)))
            return all(test)
        return NotImplemented

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

    # Parameter properties
    # ====================

    @property
    def burst_frequency_min(self):
        """Minimal bursting frequency for each gene (low expression)."""
        return self._burst_frequency[0]
    
    @burst_frequency_min.setter
    def burst_frequency_min(self, arr):
        self._burst_frequency[0] = _check_array(
            arr, 
            (self._burst_frequency.shape[1],)
        )
            

    @property
    def burst_frequency_max(self):
        """Maximal bursting frequency for each gene (high expression)."""
        return self._burst_frequency[1]
    
    @burst_frequency_max.setter
    def burst_frequency_max(self, arr):
        self._burst_frequency[1] = _check_array(
            arr, 
            (self._burst_frequency.shape[1],)
        )

    @property
    def burst_size_inv(self):
        """Inverse of average burst size for each gene."""
        return self._burst_size_inv
    
    @burst_size_inv.setter
    def burst_size_inv(self, arr):
        self._burst_size_inv = _check_array(
            arr, 
            self._burst_size_inv.shape, 
            True
        )

    @property
    def creation_rna(self):
        """mRNA creation rates."""
        return self._creation[0]
    
    @creation_rna.setter
    def creation_rna(self, arr):
        self._creation[0] = _check_array(arr, (self._creation.shape[1],))

    @property
    def creation_protein(self):
        """Protein creation rates."""
        return self._creation[1]
    
    @creation_protein.setter
    def creation_protein(self, arr):
        self._creation[1] = _check_array(arr, (self._creation.shape[1],))

    @property
    def degradation_rna(self):
        """mRNA degradation rates."""
        return self._degradation[0]
    
    @degradation_rna.setter
    def degradation_rna(self, arr):
        self._degradation[0] = _check_array(arr, (self._degradation.shape[1],))

    @property
    def degradation_protein(self):
        """Protein degradation rates."""
        return self._degradation[1]
    
    @degradation_protein.setter
    def degradation_protein(self, arr):
        self._degradation[1] = _check_array(arr, (self._degradation.shape[1],))

    @property
    def basal(self):
        """Basal expression level for each gene."""
        return self._basal
    
    @basal.setter
    def basal(self, arr):
        self._basal = _check_array(arr, self._basal.shape, True)

    @property
    def interaction(self):
        """Interactions between genes."""
        return self._interaction
    
    @interaction.setter
    def interaction(self, arr):
        self._interaction = _check_array(arr, self._interaction.shape, True)

    # Aliases (mathematical notation)
    # ===============================

    @property
    def k(self):
        """Bursting frequency bounds for each gene."""
        return self._burst_frequency
    
    @k.setter
    def k(self, arr):
        self._burst_frequency = _check_array(
            arr, 
            self._burst_frequency.shape, 
            True
        )

    @property
    def b(self):
        """Inverse of average burst size for each gene."""
        return self.burst_size_inv
    
    @b.setter
    def b(self, arr):
        self.burst_size_inv = arr

    @property
    def s(self):
        """mRNA and protein creation rates."""
        return self._creation

    @s.setter
    def s(self, arr):
        self._creation = _check_array(arr, self._creation.shape, True)

    @property
    def d(self):
        """mRNA and protein degradation rates."""
        return self._degradation
    
    @d.setter
    def d(self, arr):
        self._degradation = _check_array(arr, self._degradation.shape, True)

    @property
    def beta(self):
        """Basal expression level for each gene."""
        return self.basal
    
    @beta.setter
    def beta(self, arr):
        self.basal = arr

    @property
    def theta(self):
        """Interactions between genes."""
        return self.interaction
    
    @theta.setter
    def theta(self, arr):
        self.interaction = arr

    # Shortcut properties
    # ===================

    @property
    def c(self):
        return self.b * self.d[0] / self.s[1]

    @property
    def scale(self):
        return self.b / self.k[1]

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
            raise ValueError('n_genes should be a positive integer')
    raise TypeError('n_genes should be a positive integer')

def _check_array(arr, shape, masked = False):
    if isinstance(arr, np.ndarray):
        if arr.dtype != np.float64:
            raise ValueError(f'The argument of dtype {arr.dtype} ' 
                              'must have a np.float64 dtype')
        if arr.shape != shape:
            raise ValueError(f'The argument of shape {arr.shape} ' 
                            f'must have a {shape} shape')
        if masked:
            mask = np.ma.make_mask_none(arr.shape)
            if arr.ndim == 1:
                mask[0] = True
            else:
                mask[:, 0] = True
            
            val = 0.0
            arr[mask] = val
            arr = np.ma.array(arr, mask=mask, hard_mask=True, fill_value=val)

        return arr
    raise TypeError('The argument must be a ndarray')
    

def _masked_zeros(shape, mask):
    """Define a hard-masked array of zeros with given shape and mask."""
    return np.ma.array(
        np.zeros(shape), 
        mask=mask, 
        hard_mask=True, 
        fill_value=0.0
    )
