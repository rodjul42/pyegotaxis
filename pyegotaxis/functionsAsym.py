

# import standard libraries
import numpy as np
import numpy.linalg as algebra
import math 

def fieldv(lambda_, gamma, pos, phi ,result,bound): 
    for i in range(len(pos)):
        result[i] = field(lambda_, gamma, pos[i][0], pos[i][1], phi, bound)


def field(lambda_, gamma, x, y, phi, bound):
    '''
    ... lambda_ : [m/s] field strength
    ... R       : [m] distance from target    
    '''
    cp = math.cos(phi)
    sp = math.sin(phi)
    xx = x * cp - y * sp   
    yy = x * sp + y * cp

    Ra = math.sqrt(xx*xx+ gamma*yy*yy)
    if Ra<bound/2:
        Ra = bound/2
    return lambda_/Ra

