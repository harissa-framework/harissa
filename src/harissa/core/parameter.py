"""
Main class for network parameters
"""
import numpy as np

# Default parameter values
default_degradation_rna = np.log(2.0) / 9.0
default_degradation_protein = np.log(2.0) / 46.0
default_burst_frequency_min = 0.0 * default_degradation_rna
default_burst_frequency_max = 2.0 * default_degradation_rna
default_burst_size_inv = 0.02

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
    def __init__(self, n_genes):
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
        scale = self.scale()
        self._creation[0] = self._degradation[0] * scale
        self._creation[1] = self._degradation[1] * scale

    def __eq__(self, other):
        if isinstance(other, NetworkParameter):
            test = [other._n_genes == self._n_genes]
            for k in self._array_names():
                test.append(bool(
                    np.all(getattr(other, k) == getattr(self, k))))
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
            raise ValueError('n_genes must be strictly positive.')
    raise TypeError((f'n_genes of type {type(arg).__name__} '
                     'must be an integer.'))

def _masked_zeros(shape):
    """Array of zeros with given shape and hard-masked first column.
    Note that np.zeros is used instead of np.empty in order to avoid
    a runtime warning problem with numpy.ma operations."""
    mask = np.zeros(shape, dtype=bool)
    mask[..., 0] = True # Handle both 1D and 2D arrays
    return np.ma.array(np.zeros(shape), mask=mask, hard_mask=True)
