import numpy as np

MINUS_INF = -1e+08

def exp(x):
    if x == MINUS_INF:
        return 0
    else:
        return np.exp(x)

def log(x):
    if x == 0:
        return MINUS_INF
    else:
        return np.log(x)


def logproduct(x, y):
    if x == MINUS_INF or y == MINUS_INF:
        return MINUS_INF
    else:
        return x + y


def logsum(x, y):
    if x == -1e+08 or y == -1e+08:
        if x != -1e+08:
            return x
        else:
            return y
    elif x > y:
        return x + log(1 + np.exp(y - x))
    elif x < y:
        return y + log(1 + np.exp(x - y))


def normalize(x):
    return x / np.sum(x)


def matprint(mat, fmt="g"):
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")
