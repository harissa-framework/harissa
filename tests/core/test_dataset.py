import numpy as np
import pytest
from harissa.core import Dataset

class TestInit():
    def test(self):
        time_points = np.zeros(1)
        count_matrix = np.zeros((1, 2), dtype=np.uint)
        data = Dataset(time_points, count_matrix)

        assert data.time_points.dtype == np.float_
        assert data.count_matrix.dtype == np.uint

        assert data.time_points.shape == time_points.shape
        assert data.count_matrix.shape == count_matrix.shape

        assert data.time_points.shape[0] == data.count_matrix.shape[0]

    @pytest.mark.parametrize('times_type,count_matrix_type', [
        (None, None),
        (np.uint, np.uint),
        (np.uint, None)
    ])
    def test_wrong_type(self, times_type, count_matrix_type):
        time_points = (np.zeros(1) if times_type is None else 
                    np.zeros(1, dtype=times_type))
        count_matrix = (np.zeros((1, 2)) if count_matrix_type is None else 
                        np.zeros((1, 2), dtype=count_matrix_type))
        with pytest.raises(TypeError):
            Dataset(time_points, count_matrix)

    @pytest.mark.parametrize('times_shape,count_matrix_shape', [
        ((1,2), (1,2)),
        (1, 1),
        (1, (2,2))
    ])
    def test_wrong_shape(self, times_shape, count_matrix_shape):
        time_points = np.zeros(times_shape)
        count_matrix = np.zeros(count_matrix_shape, dtype=np.uint)
        with pytest.raises(TypeError):
            Dataset(time_points, count_matrix)

    def test_not_enough_genes(self):
        with pytest.raises(TypeError):
            Dataset(np.zeros(1), np.zeros((1, 1), dtype=np.uint))