# Perform simple data binarization using the `infer_proteins` function
import sys
import numpy as np
sys.path += ['../']

from harissa.inference import Hartree
from harissa.inference.hartree.hartree import infer_proteins
from harissa.utils import binarize

# Import raw data (run network4.py)
data = np.loadtxt('test_res_old/network4_data.txt', dtype=int, delimiter='\t')
old_bdata = np.loadtxt('test_res_old/test_binarize.txt', dtype=int, delimiter='\t')

# Store binarized data
new_data = np.zeros(data.shape, dtype='int')
new_data[:, 0] = data[:, 0] # Time points

# Calibrate the mechanistic model
a = Hartree()._get_kinetics(data)

# Get binarized values (gene-specific thresholds)
new_data[:, 1:] = infer_proteins(data, a)[:,1:].astype(int)
print(f'newdata = old_binarize(data)? {np.array_equal(new_data, old_bdata)}')

# Save binarized data
np.savetxt('test_binarize.txt', new_data, fmt='%d', delimiter='\t')

# Note: a wrapper function is available
bdata = binarize(data)
print(f'newdata = binarize(data)? {np.array_equal(new_data, bdata)}')
