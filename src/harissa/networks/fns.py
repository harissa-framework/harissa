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

    return param
