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
        # Genes plus stimulus
        G = n_genes + 1
        # Mask for ignoring stimulus
        m = np.zeros((G,G), dtype=bool)
        m[:,0] = True
        # Number of genes
        self._n_genes = n_genes
        # Initialize parameters
        self._burst_frequency = _masked_zeros((2,G), m[:2])
        self._burst_size_inv = _masked_zeros(G, m[0])
        self._creation = _masked_zeros((2,G), m[:2])
        self._degradation = _masked_zeros((2,G), m[:2])
        self._basal = _masked_zeros(G, m[0])
        self._interaction = _masked_zeros((G,G), m)
        # Default bursting parameters
        self._burst_frequency[0] = 0.0
        self._burst_frequency[1] = 2.0
        self._burst_size_inv[:] = 0.02
        # Default degradation rates
        self._degradation[0] = np.log(2.0) / 9.0
        self._degradation[1] = np.log(2.0) / 46.0
        # Default creation rates
        self._creation[0] = self._degradation[0] * self.scale
        self._creation[1] = self._degradation[1] * self.scale

    def __eq__(self, other):
        if isinstance(other, NetworkParameter):
            test = [other._n_genes == self._n_genes]
            for k in self._array_names():
                test.append(np.all(getattr(other, k) == getattr(self, k)))
            return all(test)
        return NotImplemented

    #########################
    # Network size properties
    #########################

    @property
    def n_genes(self):
        """Number of genes in the network model, without stimulus."""
        return self._n_genes

    @property
    def n_genes_stim(self):
        """Number of genes in the network model, including stimulus."""
        return self._n_genes + 1

    ######################
    # Parameter properties
    ######################

    @property
    def burst_frequency_min(self):
        """Minimal bursting frequency for each gene (low expression)."""
        return self._burst_frequency[0]

    @property
    def burst_frequency_max(self):
        """Maximal bursting frequency for each gene (high expression)."""
        return self._burst_frequency[1]

    @property
    def burst_size_inv(self):
        """Inverse of average burst size for each gene."""
        return self._burst_size_inv

    @property
    def creation_rna(self):
        """mRNA creation rates."""
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

    #################################
    # Aliases (mathematical notation)
    #################################

    @property
    def k(self):
        """Bursting frequency bounds for each gene."""
        return self._burst_frequency

    @property
    def b(self):
        """Inverse of average burst size for each gene."""
        return self._burst_size_inv

    @property
    def s(self):
        """mRNA and protein creation rates."""
        return self._creation

    @property
    def d(self):
        """mRNA and protein degradation rates."""
        return self._degradation

    @property
    def beta(self):
        """Basal expression level for each gene."""
        return self._basal

    @property
    def theta(self):
        """Interactions between genes."""
        return self._interaction

    #####################
    # Shortcut properties
    #####################

    @property
    def c(self):
        return self.b * self.d[0] / self.s[1]

    @property
    def scale(self):
        return self.b / self.k[1]

    #########
    # Methods
    #########

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
            if shallow: setattr(new_param, k, getattr(self, k))
            else: getattr(new_param, k)[:] = getattr(self, k)
        return new_param


# Masking function
def _masked_zeros(shape, mask):
    """Define a hard-masked array of zeros with given shape and mask.
    Note that np.zeros is used instead of np.empty in order to avoid
    a runtime warning problem with numpy.ma operations."""
    return np.ma.array(np.zeros(shape), mask=mask, hard_mask=True)
