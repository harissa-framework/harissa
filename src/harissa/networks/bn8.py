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

    return param