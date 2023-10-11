
import numpy as np
from scipy.interpolate import griddata,interp2d
from . import functions



def calc_lab_Drot(source,direction):
    '''Convert egocentric data to lab frame
    source    : trajectories from agents
    direction : orientatoin of agent from predict 
    '''
    if direction is None:
        return -source   #dangerous hack to make this funxtion work with and without Drot
    elif not isinstance(direction, np.ndarray):
        direction = direction[0]
    rotm =  np.array([[np.cos(direction),-np.sin(direction)],
                      [np.sin(direction),np.cos(direction)]])
    
    return -np.einsum('ij...,...j',rotm,source)

def nanquantil(data,q):
    '''Calulate quantiels for data with nans.
    nans are handelt as infinite numeber
    '''
    data_nonNaN = [i if np.isfinite(i) else np.inf for i in data]
    res = np.quantile(data_nonNaN,q)
    return res if np.isfinite(res) else np.nan  

def nanquantil_old(data,q):
    N = len(data)
    data_nonNaN = [i for i in data if np.isfinite(i)]
    N_nan = N - len(data_nonNaN)

    q_nonNaN = q + N_nan/N
    if q_nonNaN>1:
        return np.nan
    else:
        return np.quantile(data_nonNaN,q_nonNaN)

from . import functions3dDiff as functionsC

def calc_t_snr(lambda_,speed,dx,dy,snr=1,bound=0):
    '''
    Calulate time for given SNR for temporal comperrison
    '''
    return ( \
                 (  functionsC.field(1,dx,dy,bound)  * snr  )  \
              /  ( lambda_*speed**2*(functionsC.grad_field(1,dx,dy,bound)**2).sum() )                                  \
            )**(1/3) 

def calc_snr(t,lambda_,speed,dx,dy,bound=0):
    '''
    Calulate SNR for temporal comperrison
    '''
    return ( \
                 (  lambda_*speed**2*t**3*(functionsC.grad_field(1,dx,dy,bound)**2).sum()  )  \
              /  (  functionsC.field(1, dx,dy,bound)  )                                  \
            )

def calc_snr_sc(t,lambda_,agent_size,dx,dy,bound=0):
    '''
    Calulate SNR for spatial comperrison
    '''
    return ( \
                 (  lambda_*agent_size**2*t*(functionsC.grad_field(1,dx,dy,bound)**2).sum()  )  \
              /  (  functionsC.field(1,dx,dy,bound)  )                                  \
            )



def del_radial_info_2(L,sim_data):
    p = []
    for phi,kernel in sim_data.SCkernels.items():
        p.append(np.sum(kernel*L))
    Lsc = np.zeros_like(L)  
    for (phi,kernel),w in zip(sim_data.SCkernels.items(),p):
        Lsc += kernel*w        
    Lsc = np.maximum( Lsc, sim_data.minL )
    Lsc[sim_data.nogo_area] = sim_data.minL
    return Lsc / (np.sum(Lsc))


def del_radial_info(L,sim_data):
    Lphi = functions.marginal_phi(L*sim_data.nogo_mask,sim_data)/sim_data.norm
    Lsc = np.zeros_like(L)  
    for Lphi_i,(phi_i,idx) in zip(Lphi,sim_data.phi_marginal_idx.items()):
        Lsc.ravel()[idx] = Lphi_i

    Lsc = np.maximum( Lsc, sim_data.minL )
    Lsc[sim_data.nogo_area] = sim_data.minL
    return Lsc / (np.sum(Lsc))

def nanecdf(data):
    '''Emperical CDF with nanas as inf
    '''
    N = len(data)
    data_nonNaN = [i for i in data if np.isfinite(i)]
    N_nan = len(data_nonNaN)
    xs = np.sort(data_nonNaN)
    ys = np.arange(1, N_nan+.5)/float(N)
    return xs, ys

def ecdf(x):
    '''Emperical CDF 
    '''
    xs = np.sort(x)
    ys = np.arange(1, len(xs)+.5)/float(len(xs))
    return xs, ys




def calc_curv(x,y,t):
    '''Calualte curvature
    x,y : x and y compontents of trajectorie
    t   : time of x and y
    '''
    dt=t[1] - t[0]
    dx = ((x[1:] - x[:-1])/dt)[1:]
    dy = ((y[1:] - y[:-1])/dt)[1:]
    ddx= (-x[:-2]-x[2:] + 2*x[1:-1])/dt**2
    ddy= (-y[:-2]-y[2:] + 2*y[1:-1])/dt**2
    #return (ddy**2 + ddx**2)
    return (dx*ddy - dy*ddx)/(dx**2+dy**2)**(3/2)

def cart2pol(x,y):
    phi = np.arctan2(y,x)
    r = np.sqrt(x*x+y*y)
    return r,phi
