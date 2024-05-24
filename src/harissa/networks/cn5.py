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

    return param
