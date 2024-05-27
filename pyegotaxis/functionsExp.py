
# import standard libraries
import numpy as np
import numpy.linalg as algebra
import math 
from ._cfunctionsExp import *



def field(A,kappa, x, y, bound):
    '''
    ... A       : [1/s] Amplitude
    ... kappa   : [1/m] decay
    ... R       : [m] distance from target    
    '''
    R = math.sqrt(x*x+y*y)
    if (R<bound/2):
        R = bound/2
    return A*math.exp(- R * kappa)



def grad_field(A,kappa, x,  y,   bound):
    R = math.sqrt(x*x+y*y)
    if (R<bound/2):
        R = bound/2;
    res_x =  A * (-x/R*kappa) * math.exp(- R * kappa)
    res_y =  A * (-y/R*kappa) * math.exp(- R * kappa)
    return np.array([res_x,res_y])

