import numpy as np
from harissa import NetworkParameter

def bn8():
    param = NetworkParameter(8)

    param.basal[:] = -4

    param.interaction[0, 1] = 10
    param.interaction[1, 2] = 10
    param.interaction[1, 3] = 10
    param.interaction[3, 2] = -10
    param.interaction[2, 3] = -10
    param.interaction[2, 2] = 5
    param.interaction[3, 3] = 5
    param.interaction[2, 4] = 10
    param.interaction[3, 5] = 10
    param.interaction[2, 5] = -10
    param.interaction[3, 4] = -10
    param.interaction[4, 7] = -10
    param.interaction[5, 6] = -10
    param.interaction[4, 6] = 10
    param.interaction[5, 7] = 10
    param.interaction[7, 8] = 10
    param.interaction[6, 8] = -10

    param.genes_names = np.array(['', '1', '2', '3', '4', '5', '6', '7', '8'])

    param.layout = np.array([
        [-1.        , -0.75552351],
        [-0.60455711, -0.45673961],
        [-0.10688567, -0.33705544],
        [-0.35343194, -0.01072176],
        [ 0.03541208,  0.22998099],
        [ 0.23090729, -0.02877738],
        [ 0.31515515,  0.58084327],
        [ 0.64485295,  0.1444563 ],
        [ 0.83854725,  0.63353715]
    ])

    return param