import pytest
import numpy as np
from json import load, dump
from inspect import getmembers

from harissa import NetworkParameter

@pytest.fixture(scope='module')
def network_parameter():
    p = NetworkParameter(2)
    p.interaction[0, 1] = 1
    p.interaction[1, 1] = 1
    p.gene_names = np.array(['stim', '1', '2'])
    return p

@pytest.fixture(scope='module')
def network_parameter_layout():
    p = NetworkParameter(2)
    p.interaction[0, 1] = 1
    p.interaction[1, 1] = 1
    p.layout = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    return p

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

def test_setter_gene_names(network_parameter):
    with pytest.raises(TypeError):
        network_parameter.gene_names = 2

    with pytest.raises(TypeError):
        network_parameter.gene_names = np.empty((4, 2), dtype=np.str_)

    with pytest.raises(TypeError):
        network_parameter.gene_names = np.empty(
            network_parameter.n_genes, dtype=np.str_
        )

def test_setter_layout(network_parameter):
    with pytest.raises(TypeError):
        network_parameter.layout = 2

    with pytest.raises(TypeError):
        network_parameter.layout = np.empty(2)

    with pytest.raises(TypeError):
        network_parameter.layout = np.empty(
            (network_parameter.n_genes_stim, 2),
            dtype=np.str_
        )

    with pytest.raises(TypeError):
        network_parameter.layout = np.empty((network_parameter.n_genes, 2))

    with pytest.raises(TypeError):
        network_parameter.layout=np.empty((network_parameter.n_genes_stim, 4))

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
        interaction=network_parameter.interaction,
        gene_names=network_parameter.gene_names
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
    np.savetxt(path/'gene_names.txt',network_parameter.gene_names,fmt="%s")

    return path

@pytest.fixture(scope='module')
def json_file(tmp_path_factory, network_parameter):
    path = tmp_path_factory.mktemp('parameter') / 'parameter.json'
    serialized_dict = {
        'burst_frequency_min': {
            f'{i}': float(network_parameter.burst_frequency_min[i])
            for i in range(1, network_parameter.n_genes_stim)
        },
        'burst_frequency_max': {
            f'{i}': float(network_parameter.burst_frequency_max[i])
            for i in range(1, network_parameter.n_genes_stim)
        },
        'burst_size_inv':{
            f'{i}': float(network_parameter.burst_size_inv[i])
            for i in range(1, network_parameter.n_genes_stim)
        },
        'creation_rna': {
            f'{i}': float(network_parameter.creation_rna[i])
            for i in range(1, network_parameter.n_genes_stim)
        },
        'creation_protein' : {
            f'{i}': float(network_parameter.creation_protein[i])
            for i in range(1, network_parameter.n_genes_stim)
        },
        'degradation_rna': {
            f'{i}': float(network_parameter.degradation_rna[i])
            for i in range(1, network_parameter.n_genes_stim)
        },
        'degradation_protein':{
            f'{i}': float(network_parameter.degradation_protein[i])
            for i in range(1, network_parameter.n_genes_stim)
        },
        'basal':{
            f'{i}': float(network_parameter.basal[i])
            for i in range(1, network_parameter.n_genes_stim)
        },
        'interaction': {
            'stimulus -> 1' : float(network_parameter.interaction[0, 1]),
            '1 -> 1': float(network_parameter.interaction[1, 1])
        },
        'gene_names':{
            'stimulus': str(network_parameter.gene_names[0]),
            '1': str(network_parameter.gene_names[1]),
            '2': str(network_parameter.gene_names[2])
        }
    }

    with open(path, 'w') as fp:
        dump(serialized_dict, fp)

    return path

@pytest.fixture(scope='module')
def json_file_layout(tmp_path_factory, network_parameter_layout):
    path = tmp_path_factory.mktemp('parameter') / 'parameter_layout.json'
    serialized_dict = {
        'burst_frequency_min': {
            f'{i}': float(network_parameter_layout.burst_frequency_min[i])
            for i in range(1, network_parameter_layout.n_genes_stim)
        },
        'burst_frequency_max': {
            f'{i}': float(network_parameter_layout.burst_frequency_max[i])
            for i in range(1, network_parameter_layout.n_genes_stim)
        },
        'burst_size_inv':{
            f'{i}': float(network_parameter_layout.burst_size_inv[i])
            for i in range(1, network_parameter_layout.n_genes_stim)
        },
        'creation_rna': {
            f'{i}': float(network_parameter_layout.creation_rna[i])
            for i in range(1, network_parameter_layout.n_genes_stim)
        },
        'creation_protein' : {
            f'{i}': float(network_parameter_layout.creation_protein[i])
            for i in range(1, network_parameter_layout.n_genes_stim)
        },
        'degradation_rna': {
            f'{i}': float(network_parameter_layout.degradation_rna[i])
            for i in range(1, network_parameter_layout.n_genes_stim)
        },
        'degradation_protein':{
            f'{i}': float(network_parameter_layout.degradation_protein[i])
            for i in range(1, network_parameter_layout.n_genes_stim)
        },
        'basal':{
            f'{i}': float(network_parameter_layout.basal[i])
            for i in range(1, network_parameter_layout.n_genes_stim)
        },
        'interaction': {
            'stimulus -> 1': float(network_parameter_layout.interaction[0, 1]),
            '1 -> 1': float(network_parameter_layout.interaction[1, 1])
        },
        'layout':{
            'stimulus': network_parameter_layout.layout[0].tolist(),
            '1': network_parameter_layout.layout[1].tolist(),
            '2': network_parameter_layout.layout[2].tolist()
        }
    }

    with open(path, 'w') as fp:
        dump(serialized_dict, fp)

    return path

@pytest.fixture(scope='module')
def json_file_error(tmp_path_factory):
    path = tmp_path_factory.mktemp('parameter') / 'parameter_error.json'
    serialized_dict = {
        'burst_frequency_min': {'1': 0.0},
        'burst_frequency_max': {'1': 0.0, '2': 1.0},
        'burst_size_inv':{'1': 0.0, '2': 1.0},
        'creation_rna': {'1': 0.0, '2': 1.0},
        'creation_protein' : {'1': 0.0, '2': 1.0},
        'degradation_rna': {'1': 0.0, '2': 1.0},
        'degradation_protein':{'1': 0.0, '2': 1.0},
        'basal':{'1': 0.0, '2': 1.0},
        'interaction': {
            'stimulus -> 1': 1,
            '1 -> 1': 1
        }
    }

    with open(path, 'w') as fp:
        dump(serialized_dict, fp)

    return path

@pytest.fixture(scope='module')
def json_file_opt_error(tmp_path_factory):
    path = tmp_path_factory.mktemp('parameter') / 'parameter_opt_error.json'
    serialized_dict = {
        'burst_frequency_min': {'1': 0.0, '2': 1.0},
        'burst_frequency_max': {'1': 0.0, '2': 1.0},
        'burst_size_inv':{'1': 0.0, '2': 1.0},
        'creation_rna': {'1': 0.0, '2': 1.0},
        'creation_protein' : {'1': 0.0, '2': 1.0},
        'degradation_rna': {'1': 0.0, '2': 1.0},
        'degradation_protein':{'1': 0.0, '2': 1.0},
        'basal':{'1': 0.0, '2': 1.0},
        'interaction': {
            'stimulus -> 1': 1,
            '1 -> 1': 1
        },
        'gene_names': {
            'stimulus' : 'foo'
        }
    }

    with open(path, 'w') as fp:
        dump(serialized_dict, fp)

    return path

@pytest.fixture(scope='module')
def json_file_inv_inter_key_error(tmp_path_factory):
    path0=tmp_path_factory.mktemp('parameter')/'json_file_inv_inter_key_error0.json'
    path1=tmp_path_factory.mktemp('parameter')/'json_file_inv_inter_key_error1.json'
    serialized_dict = {
        'burst_frequency_min': {'1': 0.0, '2': 1.0},
        'burst_frequency_max': {'1': 0.0, '2': 1.0},
        'burst_size_inv':{'1': 0.0, '2': 1.0},
        'creation_rna': {'1': 0.0, '2': 1.0},
        'creation_protein' : {'1': 0.0, '2': 1.0},
        'degradation_rna': {'1': 0.0, '2': 1.0},
        'degradation_protein':{'1': 0.0, '2': 1.0},
        'basal':{'1': 0.0, '2': 1.0},
        'interaction': {
            'stimulus -> 1': 1,
            '1 -> foo': 1
        }
    }

    with open(path0, 'w') as fp:
        dump(serialized_dict, fp)

    serialized_dict['interaction'] = {
        'stim -> 1': 1,
        '1 -> 1': 1
    }

    with open(path1, 'w') as fp:
        dump(serialized_dict, fp)

    return path0, path1

@pytest.fixture(scope='module')
def json_file_dup_inter_key_error(tmp_path_factory):
    path=tmp_path_factory.mktemp('parameter')/'json_file_dup_inter_key_error.json'
    serialized_dict = {
        'burst_frequency_min': {'1': 0.0, '2': 1.0},
        'burst_frequency_max': {'1': 0.0, '2': 1.0},
        'burst_size_inv':{'1': 0.0, '2': 1.0},
        'creation_rna': {'1': 0.0, '2': 1.0},
        'creation_protein' : {'1': 0.0, '2': 1.0},
        'degradation_rna': {'1': 0.0, '2': 1.0},
        'degradation_protein':{'1': 0.0, '2': 1.0},
        'basal':{'1': 0.0, '2': 1.0},
        'interaction': {
            '1 -> 1': 1,
            '1  -> 1': 1
        }
    }

    with open(path, 'w') as fp:
        dump(serialized_dict, fp)

    return path

class TestIO:
    def test_load(self, npz_file, network_parameter):
        param = NetworkParameter.load(npz_file)

        assert param == network_parameter
        assert param.layout is None
        assert np.array_equal(param.gene_names, network_parameter.gene_names)

    def test_load_txt(self, txt_dir, network_parameter):
        param = NetworkParameter.load_txt(txt_dir)

        assert param == network_parameter
        assert param.layout is None
        assert np.array_equal(param.gene_names, network_parameter.gene_names)

    def test_load_json(self, json_file, network_parameter):
        param = NetworkParameter.load_json(json_file)

        assert param == network_parameter
        assert param.layout is None
        assert np.array_equal(param.gene_names, network_parameter.gene_names)

    def test_load_json_layout(self,json_file_layout,network_parameter_layout):
        param = NetworkParameter.load_json(json_file_layout)

        assert param == network_parameter_layout
        assert param.gene_names is None
        assert np.array_equal(param.layout, network_parameter_layout.layout)

    def test_load_json_error(self, json_file_error):
        with pytest.raises(RuntimeError):
            NetworkParameter.load_json(json_file_error)

    def test_load_json_opt_error(self, json_file_opt_error):
        with pytest.raises(RuntimeError):
            NetworkParameter.load_json(json_file_opt_error)

    def test_load_json_invalid_inter_key(self, json_file_inv_inter_key_error):
        with pytest.raises(RuntimeError):
            NetworkParameter.load_json(json_file_inv_inter_key_error[0])

        with pytest.raises(RuntimeError):
            NetworkParameter.load_json(json_file_inv_inter_key_error[1])

    def test_load_json_duplicate_inter_key(self,json_file_dup_inter_key_error):
        with pytest.raises(RuntimeError):
            NetworkParameter.load_json(json_file_dup_inter_key_error)

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

        for name, infos in NetworkParameter.param_names.items():
            if infos.required:
                assert np.array_equal(
                    getattr(network_parameter, name),
                    np.loadtxt(
                        (path / name).with_suffix('.txt'),
                        dtype=infos.dtype,
                        ndmin=infos.ndim
                    )
                )

    def test_save_json(self, tmp_path, network_parameter):
        path = network_parameter.save_json(tmp_path / 'foo.json')

        assert path.is_file()
        with open(path, 'r') as fp:
            serialize_dict = load(fp)
        for name, infos in NetworkParameter.param_names.items():
            if infos.required:
                assert name in serialize_dict


