## 'functions.py' = library implementing decision strategy, called by 'egotaxis.py' for Bayesian chemotaxis
# Andra Auconi, Benjamin Friedrich, 2020-2021

# import standard libraries
import numpy as np
import numpy.linalg as algebra
import math 


def xy_ij(N,space_discretization_step):
    """
    Calualte the x and y coordiantes as used for other fields
    """
    x_ij = np.zeros(shape=(N,N), dtype=float)
    y_ij = np.zeros(shape=(N,N), dtype=float)
    center = (N-1)/2
    
    for i in range(N):
        for j in range(N):            
            x_ij[i,j] = (i-center)*space_discretization_step
            y_ij[i,j] = (j-center)*space_discretization_step
            
                
    return x_ij,y_ij

def distances_ij(N,space_discretization_step):
    
    """
    Matrix of distances from center to x_ij
    i=0 corrseponds to x=-l+space_discretization_step/2
    i=N-1 corrseponds to x=l-space_discretization_step/2
    """
          
    R_ij = np.zeros(shape=(N,N), dtype=float)
    center = (N-1)/2
    
    for i in range(N):
        for j in range(N):            
            x_ij = np.array([(i-center)*space_discretization_step,(j-center)*space_discretization_step])
            R_ij[i][j] = algebra.norm(x_ij)
                
    return R_ij

# Initialization of the likelihood map [dimensionless = probability per spatial bin]
def L0(N,space_discretization_step,l,polarized = 0.003,angle=None,strenght=1e-4):
    '''
    Gaussian shape, polarization is the inverse variance with bias in angle direction
    polarized : caution, polarized = 0 gives uniform prior, but numerical problem with convection
    angle     : move Gaussian shape slightly in one direction, None -> centered [radian]
    strenght  : amount of shift from center [m]
    '''    
    if angle is None:
        angle =0
        strenght =0
    L = np.zeros(shape=(N,N), dtype=float)
    center = (N-1)/2
    
    for i in range(N):
        for j in range(N):            
            #x_ij = np.array([(i-center)*space_discretization_step,(j-center)*space_discretization_step])
            #R = algebra.norm(x_ij)
            #L[i][j] = np.exp(-polarized*np.power(R,2)/l)
            x = (i-center+strenght * np.sin(angle))*space_discretization_step
            y = (j-center+strenght * np.cos(angle))*space_discretization_step
            L[i][j] = np.exp(-polarized*(x*x+y*y)/l)
     
    L /= np.sum(L)
           
    return L

def L0_mask(mask,N,space_discretization_step,l,polarized = 0.003,angle=None,strenght=0):
    L = L0(N,space_discretization_step,l,polarized,angle,strenght)*mask
    L /= np.sum(L)
    return L


      

def entropy(L):
    S = -np.sum(np.multiply(L,np.log2(L))) # COMMENT: likelihood already represents probabilities (thus no need to multiply by bin size)
    return S

def entropy_phi(L,sim_data):
    '''Calulate directional entropy using a list of indices for each direction
    L        : Likelihood map
    sim_data : simdata class instance
    '''
    pphi = np.array([np.sum(L.ravel()[idx]) for _,idx in sim_data.phi_marginal_idx.items()])
    Sphi = -np.sum(pphi*np.log2(pphi)) # COMMENT: likelihood already represents probabilities (thus no need to multiply by bin size)
    return Sphi

def entropy_r(L,sim_data):
    '''Calulate radial entropy using a list of indices for each direction
    L        : Likelihood map
    sim_data : simdata class instance
    '''
    pr = np.array([np.sum(L.ravel()[idx]) for _,idx in sim_data.r_marginal_idx.items()])
    Sr = -np.sum(pr*np.log2(pr)) # COMMENT: likelihood already represents probabilities (thus no need to multiply by bin size)
    return Sr

def marginal_phi(L,sim_data):
    '''Calulate marginal directional distribution using a list of indices for each direction
    L        : Likelihood map
    sim_data : simdata class instance
    '''
    pphi = np.array([np.sum(L.ravel()[idx]) for _,idx in sim_data.phi_marginal_idx.items()])
    return pphi/sim_data.dphi   # calulate the denisty 

