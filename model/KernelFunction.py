import math
import numpy as np


def kernelfunction(Type, u, v, p):

    if(Type==1): # Linear kernel
        return np.dot(u,v)
    if(Type==2): # RBF kernel
        return pow(math.e,(-np.dot(u-v,u-v)/(p**2)))
