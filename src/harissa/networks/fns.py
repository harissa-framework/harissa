import numpy as np
from harissa import NetworkParameter

def fn4():
    param = NetworkParameter(4)
    
    param.basal[:] = -5
    param.interaction[0, 1] = 10
    param.interaction[1, 2] = 10
    param.interaction[1, 3] = 10
    param.interaction[3, 4] = 10
    param.interaction[4, 1] = -10
    param.interaction[2, 2] = 10
    param.interaction[3, 3] = 10

    param.genes_names = np.array(['', '1', '2', '3', '4'])

    param.layout = np.array([
        [-0.84145247, -0.63293029],
        [-0.00991027, -0.10006157],
        [ 0.70124706, -0.78540447],
        [-0.34596642,  0.80083193],
        [ 0.49608211,  0.71756441]
    ])

    return param

def fn8():
    param = NetworkParameter(8)

    param.basal[:] = -5
    param.interaction[0, 1] = 10
    param.interaction[1, 2] = 10
    param.interaction[2, 3] = 10
    param.interaction[3, 4] = 10
    param.interaction[3, 5] = 10
    param.interaction[3, 6] = 10
    param.interaction[4, 1] = -10
    param.interaction[5, 1] = -10
    param.interaction[6, 1] = -10
    param.interaction[4, 4] = 10
    param.interaction[5, 5] = 10
    param.interaction[6, 6] = 10
    param.interaction[4, 8] = -10
    param.interaction[4, 7] = -10
    param.interaction[6, 7] = 10
    param.interaction[7, 6] = 10
    param.interaction[8, 8] = 10

    param.genes_names = np.array(['', '1', '2', '3', '4', '5', '6', '7', '8'])

    param.layout = np.array([
        [-0.55941181, -0.79131586],
        [-0.06753805, -0.30039785],
        [ 0.67662283, -0.36964261],
        [ 0.28762778, -0.01938143],
        [ 0.03172841,  0.45621003],
        [ 0.28005006, -0.74265862],
        [-0.46837998,  0.07635682],
        [-0.58013184,  0.69082952],
        [ 0.3994326 ,  1.        ]
    ])

    return param
