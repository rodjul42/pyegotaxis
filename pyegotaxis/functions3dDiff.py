
# import standard libraries
import numpy as np
import numpy.linalg as algebra
import math 
from ._cfunctions import *



def field(lambda_, x, y, bound):
    '''
    ... lambda_ : [m/s] field strength
    ... R       : [m] distance from target    
    '''
    R = math.sqrt(x*x+y*y)
    if (R<bound/2):
        R = bound/2
    return lambda_/R



def grad_field(lambda_,  x,  y,   bound):
    R = math.sqrt(x*x+y*y)
    if (R<bound/2):
        R = bound/2;
    res_x =  x * -lambda_/math.pow(R,3);
    res_y =  y * -lambda_/math.pow(R,3);
    return np.array([res_x,res_y])

