

# import standard libraries
import numpy as np
import numpy.linalg as algebra
import math 

def fieldv(lambda_, vdrift, D0, pos, phi ,result,bound): 
    for i in range(len(pos)):
        result[i] = field(lambda_, vdrift, D0, pos[i][0], pos[i][1], phi, bound)


def field(lambda_, vdrift, D0, x, y, phi, bound):
    '''
    ... lambda_ : [m/s] field strength
    ... R       : [m] distance from target    
    '''
    cp = math.cos(phi)
    sp = math.sin(phi)
    xx = x * cp - y * sp   
    yy = x * sp + y * cp

    Ra = math.sqrt(xx*xx+yy*yy)
    if Ra<bound/2:
        Ra = bound/2
    return lambda_/Ra * math.exp( vdrift/(2*D0) * (  - xx - Ra) ) 
    #lambda_/(Ra*4*D0*math.pi) * math.exp( vdrift/(2*D0) * ( xx - Ra) ) 
    #lambda_/Ra * math.exp( 2*math.pi*lambda_*vdrift * ( xx - Ra) ) 
    #ll == lambda_*(4*D0*math.pi)