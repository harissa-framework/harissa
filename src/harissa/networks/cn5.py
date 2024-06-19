import numpy as np
from harissa import NetworkParameter

def cn5():
    param = NetworkParameter(5)

    
    param.basal[1:] = [-5, 4, 4, -5, -5]
    param.interaction[0, 1] = 10
    param.interaction[1, 2] = -10
    param.interaction[2, 3] = -10
    param.interaction[3, 4] = 10
    param.interaction[4, 5] = 10
    param.interaction[5, 1] = -10

    param.genes_names = np.array(['', '1', '2', '3', '4', '5'])

    param.layout = np.array([
        [-1.        , -0.7402784 ],
        [-0.37545312, -0.27792309],
        [ 0.41538345, -0.49134294],
        [ 0.8938372 ,  0.16086717],
        [ 0.41486746,  0.80790389],
        [-0.34863499,  0.54077337]
    ])

    return param
