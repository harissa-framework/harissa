import numpy as np
from harissa.core import Dataset
from harissa.processing import binarize


def test_binarize():
    time_points = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
    count_matrix = np.array([
        # s g1 g2 g3
        [0, 4, 1, 0], # Cell 1
        [0, 5, 0, 1], # Cell 2
        [1, 1, 2, 4], # Cell 3
        [1, 2, 0, 8], # Cell 4
        [1, 0, 0, 3], # Cell 5
    ], dtype=np.uint)

    dataset = binarize(Dataset(time_points, count_matrix))

    assert np.array_equal(dataset.time_points, time_points)
    assert dataset.gene_names is None

    assert np.all((dataset.count_matrix == 0) | (dataset.count_matrix == 1))

    gene_names = np.array(['s', 'g1', 'g2', 'g3'], dtype=np.str_)
    dataset = binarize(Dataset(time_points, count_matrix, gene_names))

    assert np.array_equal(dataset.time_points, time_points)
    assert np.array_equal(dataset.gene_names, gene_names)
    assert np.all((dataset.count_matrix == 0) | (dataset.count_matrix == 1))
