import numpy as np
import anndata as ad
import pytest
from harissa.core import Dataset

@pytest.fixture(scope='module')
def time_points():
    return np.array([0.0, 0.0, 1.0, 1.0, 1.0])

@pytest.fixture(scope='module')
def count_matrix():
    return np.array([
        # s g1 g2 g3
        [0, 4, 1, 0], # Cell 1
        [0, 5, 0, 1], # Cell 2
        [1, 1, 2, 4], # Cell 3
        [1, 2, 0, 8], # Cell 4
        [1, 0, 0, 3], # Cell 5
    ], dtype=np.uint)

@pytest.fixture(scope='module')
def gene_names(count_matrix):
    return np.array([f'g{i}' for i in range(count_matrix.shape[1])])

@pytest.fixture
def dataset(time_points, count_matrix) -> Dataset:
    return Dataset(time_points, count_matrix)

@pytest.fixture
def dataset_with_gene_names(time_points, count_matrix, gene_names) -> Dataset:
    return Dataset(time_points, count_matrix, gene_names)

@pytest.fixture
def adata(time_points, count_matrix) -> ad.AnnData:
    return ad.AnnData(count_matrix, {'time_points': time_points})

@pytest.fixture
def adata_with_gene_names(time_points, count_matrix, gene_names) -> ad.AnnData:
    d = ad.AnnData(count_matrix, {'time_points': time_points})
    d.var_names = gene_names
    return d

@pytest.fixture(scope='module')
def npz_file(tmp_path_factory, time_points, count_matrix):
    path = tmp_path_factory.mktemp('dataset') / 'dataset.npz'
    np.savez_compressed(
        path, 
        time_points=time_points, 
        count_matrix=count_matrix
    )
    return path

@pytest.fixture(scope='module')
def txt_dir(tmp_path_factory, time_points, count_matrix):
    path = tmp_path_factory.mktemp('dataset') / 'dataset'
    path.mkdir()
    np.savetxt(path / 'time_points.txt', time_points)
    np.savetxt(path / 'count_matrix.txt', count_matrix)
    return path

@pytest.fixture(scope='module')
def txt_file(tmp_path_factory, time_points, count_matrix):
    path = tmp_path_factory.mktemp('dataset') / 'dataset.txt'

    data = count_matrix.copy()
    data[:, 0] = time_points.astype(np.uint)

    np.savetxt(path, data)
    return path

@pytest.fixture(scope='module')
def hd5ad_file(tmp_path_factory, time_points, count_matrix):
    path = tmp_path_factory.mktemp('dataset') / 'dataset.h5ad'
    adata = ad.AnnData(count_matrix, obs={'time_points': time_points})
    adata.write_h5ad(path)

    return path

# @pytest.fixture(scope='module')
# def zarr_file(tmp_path_factory, time_points, count_matrix):
#     path = tmp_path_factory.mktemp('dataset') / 'dataset.zarr'
#     adata = ad.AnnData(count_matrix, obs={'time_points': time_points})
#     adata.write_zarr(path)

#     return path

def test_as_dict(
        dataset, 
        dataset_with_gene_names, 
        time_points, 
        count_matrix, 
        gene_names
    ):
    data = dataset.as_dict()

    assert len(data) == 2
    assert 'time_points' in data
    assert 'count_matrix' in data

    assert np.array_equal(data['time_points'], time_points)
    assert np.array_equal(data['count_matrix'], count_matrix)

    data = dataset_with_gene_names.as_dict()
    
    assert len(data) == 3
    assert 'time_points' in data
    assert 'count_matrix' in data
    assert 'gene_names' in data

    assert np.array_equal(data['time_points'], time_points)
    assert np.array_equal(data['count_matrix'], count_matrix)
    assert np.array_equal(data['gene_names'], gene_names)


def test_as_annData(dataset: Dataset):
    adata = dataset.as_annData()

    assert np.array_equal(adata.X, dataset.count_matrix)
    assert np.array_equal(adata.obs['time_points'], dataset.time_points)
    assert np.array_equal(
        adata.obs_names, 
        np.array([f'Cell_{i+1}' for i in range(adata.n_obs)])
    )
    assert np.array_equal(
        adata.var_names, 
        np.array([f'Gene_{i}' for i in range(adata.n_vars)])
    )

def test_as_annData_with_gene_names(dataset_with_gene_names: Dataset):
    adata = dataset_with_gene_names.as_annData()

    assert np.array_equal(adata.X, dataset_with_gene_names.count_matrix)
    assert np.array_equal(
        adata.obs['time_points'], 
        dataset_with_gene_names.time_points
    )
    assert np.array_equal(
        adata.obs_names, 
        np.array([f'Cell_{i+1}' for i in range(adata.n_obs)])
    )
    assert np.array_equal(adata.var_names, dataset_with_gene_names.gene_names)

def test_from_anndata(adata: ad.AnnData):
    dataset = Dataset.from_annData(adata)

    assert np.array_equal(adata.X, dataset.count_matrix)
    assert np.array_equal(adata.obs['time_points'], dataset.time_points)

def test_from_anndata_with_gene_names(adata_with_gene_names: ad.AnnData):
    dataset = Dataset.from_annData(adata_with_gene_names)

    assert np.array_equal(adata_with_gene_names.X, dataset.count_matrix)
    assert np.array_equal(
        adata_with_gene_names.obs['time_points'], 
        dataset.time_points
    )
    assert np.array_equal(adata_with_gene_names.var_names, dataset.gene_names)

def test_from_anndata_missing_time_points(count_matrix):
    adata = ad.AnnData(count_matrix)

    with pytest.raises(RuntimeError):
        Dataset.from_annData(adata)

class TestInit():
    def test(self, time_points, count_matrix):
        data = Dataset(time_points, count_matrix)

        assert data.time_points.dtype == np.float64
        assert data.count_matrix.dtype == np.uint

        assert data.time_points.shape == time_points.shape
        assert data.count_matrix.shape == count_matrix.shape

        assert data.time_points.shape[0] == data.count_matrix.shape[0]

    @pytest.mark.parametrize('times_type,count_matrix_type,gene_names_type', [
        (None, None, None),
        (np.uint, np.uint, None),
        (np.uint, None, None),
        (None, np.uint, np.float64)
    ])
    def test_wrong_type(self, times_type, count_matrix_type, gene_names_type):
        time_points = (np.zeros(1) if times_type is None else 
                    np.zeros(1, dtype=times_type))
        count_matrix = (np.zeros((1, 2)) if count_matrix_type is None else 
                        np.zeros((1, 2), dtype=count_matrix_type))
        gene_names = (np.zeros(2, dtype=gene_names_type) 
                      if gene_names_type is not None else None) 
        with pytest.raises(TypeError):
            Dataset(time_points, count_matrix, gene_names)

    @pytest.mark.parametrize('times_shape,count_matrix_shape,gene_names_shape', 
    [
        ((1,2), (1,2), 2),
        (1, 1, 2),
        (1, (2,2), 2),
        (1, (1,2), 1),
    ])
    def test_wrong_shape(self, 
        times_shape, 
        count_matrix_shape, 
        gene_names_shape
    ):
        time_points = np.zeros(times_shape)
        count_matrix = np.zeros(count_matrix_shape, dtype=np.uint)
        gene_names = np.full(gene_names_shape, 'foo', dtype=np.str_)
        with pytest.raises(TypeError):
            Dataset(time_points, count_matrix, gene_names)

    def test_not_enough_genes(self):
        with pytest.raises(TypeError):
            Dataset(np.zeros(1), np.zeros((1, 1), dtype=np.uint))

class TestIO:

    def test_load(self, npz_file, time_points, count_matrix):
        dataset = Dataset.load(npz_file)

        assert np.array_equal(dataset.time_points, time_points)
        assert np.array_equal(dataset.count_matrix, count_matrix)


    def test_load_txt(self, txt_dir, time_points, count_matrix):
        dataset = Dataset.load_txt(txt_dir)
        
        assert np.array_equal(dataset.time_points, time_points)
        assert np.array_equal(dataset.count_matrix, count_matrix)

    def test_load_txt_backward_compatibility(self, 
        txt_file, 
        time_points, 
        count_matrix
    ):
        dataset = Dataset.load_txt(txt_file)

        assert np.array_equal(dataset.time_points, np.floor(time_points))
        assert np.array_equal(dataset.count_matrix, count_matrix)

    def test_load_txt_backward_compatibility_notfound(self, tmp_path):
        with pytest.raises(RuntimeError):
            Dataset.load_txt(tmp_path / 'foo.txt')

    def test_load_h5ad(self, hd5ad_file, time_points, count_matrix):
        dataset = Dataset.load_h5ad(hd5ad_file)

        assert np.array_equal(dataset.time_points, time_points)
        assert np.array_equal(dataset.count_matrix, count_matrix)

    # def test_load_zarr(self, zarr_file, time_points, count_matrix):
    #     dataset = Dataset.load_zarr(zarr_file)

    #     assert np.array_equal(dataset.time_points, time_points)
    #     assert np.array_equal(dataset.count_matrix, count_matrix)

    def test_save_txt(self, tmp_path, dataset):
        path = dataset.save_txt(tmp_path / 'dataset')
        data = { 
            'time_points': np.loadtxt(path/'time_points.txt'),
            'count_matrix': np.loadtxt(path/'count_matrix.txt', dtype=np.uint)
        }

        assert np.array_equal(data['time_points'], dataset.time_points)
        assert np.array_equal(data['count_matrix'], dataset.count_matrix)

    def test_save(self, tmp_path, dataset):
        path = dataset.save(tmp_path / 'dataset.npz')
        data = np.load(path)

        assert np.array_equal(data['time_points'], dataset.time_points)
        assert np.array_equal(data['count_matrix'], dataset.count_matrix)

    def test_save_h5ad(self, tmp_path, dataset):
        path = dataset.save_h5ad(tmp_path / 'dataset.h5ad')
        data = ad.read_h5ad(path)

        assert np.array_equal(
            np.array(data.obs['time_points']), 
            dataset.time_points
        )
        assert np.array_equal(data.X, dataset.count_matrix)

    # def test_save_zarr(self, tmp_path, dataset):
    #     path = dataset.save(tmp_path / 'dataset.zarr')
    #     data = ad.read_zarr(path)

    #     assert np.array_equal(
    #         np.array(data.obs['time_points']), 
    #         dataset.time_points
    #     )
    #     assert np.array_equal(data.X, dataset.count_matrix)