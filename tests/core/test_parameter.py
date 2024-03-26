import pytest
import numpy as np
from inspect import getmembers

from harissa import NetworkParameter

@pytest.fixture(scope='module')
def network_parameter():
    return NetworkParameter(1)

class TestInit:
    def test_init_neg(self):
        with pytest.raises(ValueError):
            NetworkParameter(0)

        with pytest.raises(ValueError):
            NetworkParameter(-1)

    def test_init_not_int(self):
        with pytest.raises(TypeError):
            NetworkParameter(2.5)

    def test_init(self):
        n_genes = 1
        param = NetworkParameter(n_genes)

        assert param.n_genes == n_genes
        assert param.n_genes_stim == n_genes + 1
        
        for array_name in param._array_names():
            array = getattr(param, array_name)

            assert array.T.shape[0] == param.n_genes_stim
            assert np.all(array.mask[..., 0])

        assert param.interaction.shape[0] == param.n_genes_stim
    

def test_properties(network_parameter):

    props = [
        (k, getattr(network_parameter, k)) 
        for k, _ in getmembers(
            type(network_parameter), 
            lambda o: isinstance(o, property)
        )
    ]

    for _, prop in filter(lambda p:isinstance(p[1], np.ma.MaskedArray), props):
        assert np.all(
            np.fromiter(
                ([e is np.ma.masked for e in np.atleast_1d(prop[..., 0])]),
                bool
            )
        )
    
    assert np.all(
        np.fromiter(
            ([e is np.ma.masked 
              for e in np.atleast_1d(network_parameter.c()[..., 0])]),
            bool
        )
    )
    
    for _, prop in filter(lambda p:isinstance(p[1], int), props):
        assert prop > 0 

def test_setters(network_parameter):

    with pytest.raises(AttributeError):
        network_parameter.basal = 2
    
    with pytest.raises(AttributeError):
        network_parameter.basal = np.empty(network_parameter.basal.shape)

    network_parameter.basal[:] = 1.0

    assert network_parameter.basal[0] is np.ma.masked
    assert network_parameter.basal[1] == 1.0

    network_parameter.basal[:] = np.full(network_parameter.basal.shape, 2.0)

    assert network_parameter.basal[0] is np.ma.masked
    assert network_parameter.basal[1] == 2.0

def test_equal():
    p1 = NetworkParameter(1)
    p2 = NetworkParameter(1)

    assert p1 == p2

    with pytest.raises(NotImplementedError):
        p1 == 1

    p2.basal[1] = p1.basal[1] + 1

    assert p1 != p2


def test_copy():
    p1 = NetworkParameter(1)
    p2 = p1.copy()

    assert p1.basal[1] == p2.basal[1]

    p2.basal[1] = p1.basal[1] + 1

    assert p1.basal[1] != p2.basal[1]
    
def test_copy_shallow():
    p1 = NetworkParameter(1)
    p2 = p1.copy(shallow=True)

    assert p1.basal[1] == p2.basal[1]

    p2.basal[1] = p1.basal[1] + 1

    assert p1.basal[1] == p2.basal[1]

@pytest.fixture(scope='module')
def npz_file(tmp_path_factory, network_parameter):
    path = tmp_path_factory.mktemp('parameter') / 'parameter.npz'
    np.savez_compressed(
        path,
        burst_frequency_min = network_parameter.burst_frequency_min,
        burst_frequency_max = network_parameter.burst_frequency_max,
        burst_size_inv = network_parameter.burst_size_inv,
        creation_rna= network_parameter.creation_rna,
        creation_protein = network_parameter.creation_protein,
        degradation_rna= network_parameter.degradation_rna,
        degradation_protein=network_parameter.degradation_protein,
        basal=network_parameter.basal,
        interaction=network_parameter.interaction
    )
    return path

@pytest.fixture(scope='module')
def txt_dir(tmp_path_factory, network_parameter):
    path = tmp_path_factory.mktemp('parameter') / 'parameter'
    path.mkdir()

    np.savetxt(path/'burst_frequency_min.txt',network_parameter.burst_frequency_min)
    np.savetxt(path/'burst_frequency_max.txt',network_parameter.burst_frequency_max)
    np.savetxt(path/'burst_size_inv.txt',network_parameter.burst_size_inv)
    np.savetxt(path/'creation_rna.txt',network_parameter.creation_rna)
    np.savetxt(path/'creation_protein.txt',network_parameter.creation_protein)
    np.savetxt(path/'degradation_rna.txt',network_parameter.degradation_rna)
    np.savetxt(path/'degradation_protein.txt',network_parameter.degradation_protein)
    np.savetxt(path/'basal.txt',network_parameter.basal)
    np.savetxt(path/'interaction.txt',network_parameter.interaction)

    return path

class TestIO:
    def test_load(self, npz_file, network_parameter):
        param = NetworkParameter.load(npz_file)
        
        assert param == network_parameter

    def test_load_txt(self, txt_dir, network_parameter):
        param = NetworkParameter.load_txt(txt_dir)
        
        assert param == network_parameter
    def test_save(self, tmp_path, network_parameter):

        path = network_parameter.save(tmp_path / 'foo.npz')
        data = np.load(path)

        for name, infos in NetworkParameter.param_names.items():
            if infos.required:
                assert np.array_equal(
                    getattr(network_parameter, name),
                    data[name]
                )

    def test_save_txt(self, tmp_path, network_parameter):

        path = network_parameter.save_txt(tmp_path / 'foo')

        for name, (required, dtype, ndim) in NetworkParameter.param_names.items():
            if required:
                assert np.array_equal(
                    getattr(network_parameter, name), 
                    np.loadtxt(
                        (path / name).with_suffix('.txt'), 
                        dtype=dtype, 
                        ndmin=ndim
                    )
                )