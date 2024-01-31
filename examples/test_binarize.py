# Perform simple data binarization using the `infer_proteins` function
import numpy as np
from harissa.inference import Hartree
from harissa.inference.hartree.base import infer_proteins
from harissa.utils import binarize
from harissa.utils.npz_io import load_dataset

# Import raw data (run network4.py)
dataset = load_dataset('network4_data.txt')

# Store binarized data
new_data = np.zeros(dataset.count_matrix.shape, dtype=np.uint)

# Calibrate the mechanistic model
a = Hartree()._get_kinetics(dataset, np.unique(dataset.time_points))

# Get binarized values (gene-specific thresholds)
new_data = infer_proteins(dataset, a).astype(np.uint)

# Save binarized data
np.savetxt('test_binarize.txt', new_data, fmt='%d', delimiter='\t')

# Note: a wrapper function is available
bdataset = binarize(dataset)
print('newdata = binarize(data)?'
     f' {np.array_equal(new_data, bdataset.count_matrix)}')
