# TODO: profiling 
# Ideas for speeing-up code
# - different time steps for update and prediction step
# - adaptive time step dt~1/r

# import python libraries
import numpy as np
import json
import logging
from scipy import stats

logger = logging.getLogger(__name__)

from .functions import *
from .functions3dDiff import distances_ij
from . import functionsExp
from . import functions3dDiff
from . import tools
from . import utils
from .tools import SIMLOG,SIMLOG_tcsc
class simdata():
    """
    A class for all the parameters and precomputed matrices
    """
    def __init__(self,seed=42,sampling_interval=100, sampling2_interval=100, dt=5e-3,maxTime=100,N=200,l=1,boundary='a',
            D=0, Drot=0,lambda_=1, agent_size = 0.00025, speed = 0.01,lambdaReal_=None,gammaReal=None,vdrift=None,D0=None,agent_sizeReal=None,DrotReal=None,
            target_size=None, source =  np.array([0,0.2]), field="3Ddiff", fieldReal=None,**args):
        """
        seed        : seed
        sampling_interval  : for saving results
        sampling2_interval : for saving liklyhood every sampling2_interval*sampling_interval
        dt          : [s] time step 
        maxTime     : [s] maximal integration time 
        l           : [m] side length of square simulation domain 
                       spatial domain: [-l,+l] x [-l,+l]
                       grid points correspond to center of spatial bins: -l+dl/2, ..., +l-dl/2
        N           : [] space discretization: number of spatial bins per dimension
        D           : [m^2/s] motility noise: translational diffusion coefficient
        Drot        : [radian^2/s] rotation noise: rotation diffusion coefficient
        lambda_     : [m/s] field strength, 'binding rate' = lambda_ / distance
        agent_size  : [m] radius 'a' of agent (Note: should be in range 0.0005-0.001 for reliable results)
        speed       : [m/s] speed of agent
        target_size : [m] radius of target
        source      : [m] initial position of the source
        field       : [1/s] concentraion field 3Ddiff (~1/r) or Exp (~ e**r)
        [Drot,lambda,agent_size,field]Real : paramters for the environment, if they should differ from the agent's assumption 
        """

        self.use_cfunc = True
        self.seed = seed
        np.random.seed(self.seed)
        
        # Model parameters ------------------------------------------------------------
        self.lambda_ = lambda_ # [m/s] field strength, 'binding rate' = lambda_ / distance
        self.speed = speed # [m/s] speed of agent

        # D = 1.5e-4 #
        self.D = D 
        self.Drot = Drot 

        self.agent_size = agent_size
        self.agent_size_2 = agent_size*agent_size

        if lambdaReal_ is None:
            lambdaReal_ = lambda_
        self.lambdaReal_ = lambdaReal_
        
        if agent_sizeReal is None:
            agent_sizeReal = agent_size
        self.agent_sizeReal = agent_sizeReal

        if DrotReal is None:
            DrotReal = Drot
        self.DrotReal = DrotReal
        # Initial conditions
        self.source = source

        self.dt = dt
        self.N = N
        self.T = maxTime
        self.l  = l
        self.minL = 1e-30/N**2 # [] lowest allowed likelihood (for numerical stability)
        self.boundary = boundary
        
        self.space_discretization_step = 2*l/N # [m] size of spatial bins
        max_dist = self.l - self.space_discretization_step*3 # [m] domain size: maximal allowed distance of agent from target
        self.target_size = self.space_discretization_step*4  #target_size : [m] radius of target
        

        
        if lambdaReal_ != lambda_:
                ratio = lambda_/lambdaReal_
                self.l  *= ratio
                self.space_discretization_step = 2*self.l/self.N # [m] size of spatial bins
        self.max_dist = self.l - self.space_discretization_step*3 # [m] domain size: maximal allowed distance of agent from target
        
        # Simulation parameters
        self.sampling_interval = sampling_interval # [] for saving results
        self.sampling2_interval = sampling2_interval # [] for saving liklyhood every sampling2_interval*sampling_interval
        self.phi0 = np.nan

        self.bound = 0#1.5*max(self.space_discretization_step,self.agent_size) # [m] minimal distance up to which field is computed; for numerical stability
     

        self.profile = False
        self.max_dist = max_dist
        A, k = self.calc_parameterExp(lambdaReal_, self.target_size, self.max_dist)
        self.AReal = A
        self.kReal = k
        A, k = self.calc_parameterExp(lambda_, self.target_size, self.max_dist)
        self.A = A
        self.k = k
        self.fieldReal = fieldReal
        if fieldReal == 'Asym':
            if gammaReal is None:
                raise ValueError('no value for gammaReal ')
            self.gammaReal = gammaReal
        elif fieldReal == 'Drift':
            if D0 is None or vdrift  is None:
                    raise ValueError('no value for D0 or  ')
            self.D0 = D0 
            self.vdrift = vdrift  

        self.set_receptors()
        self.field = field
        if fieldReal is None:
            fieldReal = field
        self.field_real = fieldReal
        self.calc_matrices()    


    @classmethod
    def from_simdata(cls, simlog):
        simdata = cls(**simlog.parameter)
        simdata.phi0 = simlog.parameter['phi0']
        simdata.Linit = simlog.initL
        return simdata

    @classmethod
    def from_dict(cls, simdict,initL=None):
        simdata = cls(**simdict)
        if initL is not None:
            simdata.phi0 = simdict.parameter['phi0']
            simdata.Linit = initL
        return simdata

    def _asdict(self):
        data={}
        data['k'] = self.k
        data['A'] = self.A
        data['field'] = self.field
        data['fieldReal'] = self.fieldReal
        data['seed'] = self.seed
        data['phi0'] = self.phi0
        data['maxTime'] = self.T
        data['dt'] = self.dt
        data['N'] = self.N
        data['l'] = self.l
        data['boundary'] = self.boundary
        data['D'] = self.D
        data['Drot'] = self.Drot
        data['DrotReal'] = self.DrotReal
        data['lambda_'] = self.lambda_
        data['lambdaReal_'] = self.lambdaReal_
        if self.fieldReal == 'Asym':
            data['gammaReal'] = self.gammaReal
        if self.fieldReal == 'Drift':
            data['D0'] = self.D0
            data['vdrift'] = self.vdrift
        data['agent_size'] = self.agent_size
        data['agent_sizeReal'] = self.agent_sizeReal
        data['speed'] = self.speed
        data['target_size'] = self.target_size
        data['source'] = list(self.source)
        data['N_angles'] = self.N_angles
        data['bound']  = self.bound       
        data['max_dist']  = self.max_dist       
        data['minL']  = self.minL       
        data['sampling_interval']  = self.sampling_interval  
        data['sampling2_interval']  = self.sampling2_interval  

        return data

    def __repr__(self):
        return json.dumps(self._asdict(), indent = 2)
 
    def __str__(self):
        return self.__repr__()


    def calc_parameterExp(self,lam, r1, r2):
        A = lam * r1**(-1 + r1/(r1 - r2)) * (1/r2)**(r1/(r1 - r2))
        k = np.log(r1 / r2) / (r1 - r2)
        return A, k


    def set_receptors(self,number=50):
        '''
        initialization of the vector of receptors placed equidistantly around a circle
        number : number of receptors
        '''
        self.N_angles = number # [] number of receptors (Note: local rates of binding are normalized, such that total rate of binding is independent of 'N_angles')
        # TODO: find good value for 'N_angles'
        # list of angles for receptor positions [rad]
        angles = [i*2*np.pi/self.N_angles for i in range(self.N_angles)]
        # list of corresponding unit vectors for all receptor positions
        self.directions = np.array([np.array([np.cos(phi),np.sin(phi)]) for phi in angles])


# Initialize simulation -------------------------------------------------------
    def calc_matrices(self):
        '''
        # NxN matrices of important quantities (pre-computed for all bin centers) ....        
        '''
        N = self.N
        space_discretization_step = self.space_discretization_step
        l = self.l
        self.center = ((N-1)/2)*space_discretization_step

        self.R_x_ij = np.zeros((self.N,self.N))
        distances_ij(self.R_x_ij,self.center,self.space_discretization_step)
        if self.field=='3Ddiff':
            self.r_x_ij = np.zeros((self.N,self.N))
            functions3dDiff.field_ij(self.r_x_ij,self.lambda_,self.center,self.space_discretization_step)
            self.grad_r_x_ij = np.zeros((self.N,self.N,2))
            functions3dDiff.grad_field_ij(self.grad_r_x_ij,self.lambda_,self.center,self.space_discretization_step)

            self.grad_Laplace_r_x_ij = np.zeros((self.N,self.N,2))
            functions3dDiff.grad_laplace_field_ij(self.grad_Laplace_r_x_ij,self.lambda_,self.center,self.space_discretization_step)

            H_ij = np.zeros((self.N,self.N,4))
            self.Laplace_r_x_ij = np.zeros((self.N,self.N))
            functions3dDiff.hess_field_ij(H_ij,self.Laplace_r_x_ij,self.lambda_,self.center,self.space_discretization_step)
            self.H_r_x_ij = H_ij.reshape(self.N,self.N,2,2)
        elif self.field=='Exp':
            self.r_x_ij = np.zeros((self.N,self.N))
            functionsExp.field_ij(self.r_x_ij,self.A,self.k,self.center,self.space_discretization_step)
            self.grad_r_x_ij = np.zeros((self.N,self.N,2))
            functionsExp.grad_field_ij(self.grad_r_x_ij,self.A,self.k,self.center,self.space_discretization_step)

            self.grad_Laplace_r_x_ij = np.zeros((self.N,self.N,2))
            functionsExp.grad_laplace_field_ij(self.grad_Laplace_r_x_ij,self.A,self.k,self.center,self.space_discretization_step)

            H_ij = np.zeros((self.N,self.N,4))
            self.Laplace_r_x_ij = np.zeros((self.N,self.N))
            functionsExp.hess_field_ij(H_ij,self.Laplace_r_x_ij,self.A,self.k,self.center,self.space_discretization_step)
            self.H_r_x_ij = H_ij.reshape(self.N,self.N,2,2)
        else:
            raise NotImplementedError

        # TODO: maybe we can combine certain fields into one (e.g., r_finite = total rate averaged over circumference)
        # TODO: maybe use linearized update equation by Ben [Eq. (3) in version from 08.10.2021]

        self.log_r_x_ij = np.log(self.r_x_ij)
        # total binding rate (averaged over circumference of agent) for all possible positions of source
        self.rfinite_x_ij = self.r_x_ij + (self.agent_size/2)**2*(self.H_r_x_ij[:,:,0,0]+self.H_r_x_ij[:,:,1,1])
        self.log_rfinite_x_ij = np.log(self.rfinite_x_ij)
        self.grad_rfinite_x_ij = self.grad_r_x_ij + (self.agent_size/2)**2*self.grad_Laplace_r_x_ij
        x_ij,y_ij = xy_ij(N,space_discretization_step)
        r,self.phi = utils.cart2pol(x_ij,y_ij )

        self.phi = np.arctan2(y_ij,x_ij)
        # initialize likelihood map [dimensionless = probability per spatial bin]
        self.nogo_area = self.R_x_ij>self.max_dist
        self.nogo_mask = self.R_x_ij<self.max_dist
        self.Linit = L0_mask(self.nogo_mask,N,space_discretization_step,l) # CAUTION: not uniform, parameter inside sub-function
        #self.nogo_mask2 = self.R_x_ij>0.25
        #self.phi_r = np.pi*np.where(self.nogo_mask,self.R_x_ij/self.max_dist,np.zeros_like(self.R_x_ij))
        self.clalc_list_marginal(r,self.phi,dr=self.space_discretization_step,dphi=np.pi/50)

        
        self.SCmask = np.ones_like(self.R_x_ij)
        self.SCmask[self.R_x_ij < 0.1] = 0
        self.SCmask[self.R_x_ij > self.max_dist] = 0
        self.norm = marginal_phi(np.ones_like(self.Linit)*self.SCmask,self)
        self.target_area = self.R_x_ij<self.target_size

        ###just temp
        kernelsma = {}
        eps = np.pi/50
        N=np.zeros_like(self.R_x_ij)
        #for phi0 in np.arange(-np.pi,np.pi-eps/2,eps):
        for phi0 in np.linspace(-np.pi,np.pi-2*np.pi/100,100):
            kappa = 200
            rv = stats.vonmises(kappa,loc=phi0)
            w = np.ma.masked_array(rv.pdf(self.phi) , self.R_x_ij>=self.max_dist) 
            kernelsma[phi0] = w 
            N += kernelsma[phi0] 

        for phi,kernel in kernelsma.items():
            kernel /= N
            kernel.fill_value = 0

        self.SCkernels = {phi:kernel.filled() for phi,kernel in kernelsma.items()}

    def clalc_list_marginal(self, r, phi, dr, dphi):
        '''calulate the list of indices for the calulation of directinal and radial entropy 
        r   : distances to each point of the likelihood
        phi : angles to each point of the likelihood
        dr  : resultution of radial entropy
        dphi : resultution of directinal entropy
        '''
        self.dphi = dphi
        self.dr = dr
        idx = np.arange(self.N*self.N,dtype=int)
        self.r_marginal_idx = {}
        for i in np.arange(np.min(r),self.max_dist,dr):
            self.r_marginal_idx[i] = idx[np.logical_and(r.ravel()>=i,r.ravel()<(i+dr))]
        self.drs = np.array(list(self.r_marginal_idx.keys())) + dr/2

        self.phi_marginal_idx = {}
        rmask = r.ravel()<=self.max_dist
        for i in np.arange(-np.pi,np.pi,dphi):
            phibool = np.logical_and(phi.ravel()>=i,phi.ravel()<(i+dphi))
            self.phi_marginal_idx[i] = idx[np.logical_and(phibool,rmask)]
        self.dphis = np.array(list(self.phi_marginal_idx.keys())) +dphi/2


from scipy.stats import norm
class simdata_lambda(simdata):
    def __init__(self, lam_sigma,seed=42, sampling_interval=100, sampling2_interval=100, dt=0.005, maxTime=100, N=200, l=1, boundary='a', D=0, Drot=0, lambda_=1, agent_size=0.00025, speed=0.01, target_size=None, source=np.array([0, 0.2])):
        self.lam_sigma = lam_sigma
        lambdas = np.linspace(-6*lam_sigma + lambda_,6*lam_sigma + lambda_,1001)
        dl = lambdas[1] - lambdas[0]
        self.lambdas = lambdas[lambdas-dl>0]
        self.Plambdas = norm.pdf( (self.lambdas - lambda_)/lam_sigma )*dl/lam_sigma
        self.Plambdas[0] = 1 - np.sum(self.Plambdas[1:]) #put all probalility from negative lambdas to smalles lambda

        super().__init__(seed, sampling_interval, sampling2_interval, dt, maxTime, N, l, boundary, D, Drot, lambda_, agent_size, speed, target_size, source)


    def _asdict(self):  
        data = super()._asdict()
        data['lam_sigma'] = self.lam_sigma
        return data



    def calc_matrices(self):
        '''
        # NxN matrices of important quantities (pre-computed for all bin centers) ....        
        '''
        super().calc_matrices()
        N = self.N
        space_discretization_step = self.space_discretization_step
        l = self.l
        
        bound = self.bound
        self.center = ((N-1)/2)*space_discretization_step
        
        
        self.r_x_ij = np.zeros((self.N,self.N))
        self.grad_r_x_ij = np.zeros((self.N,self.N,2))
        self.grad_Laplace_r_x_ij = np.zeros((self.N,self.N,2))
        self.Laplace_r_x_ij = np.zeros((self.N,self.N))
        H_ij = np.zeros((self.N,self.N,4))
        
        tmp_1d = np.zeros((self.N,self.N))
        tmp_2d = np.zeros((self.N,self.N,2))
        tmp_4d = np.zeros((self.N,self.N,4))
        for lam,Plam in zip(self.lambdas,self.Plambdas):
            tmp_1d *=0
            functionsC.field_ij(tmp_1d,lam,self.center,self.space_discretization_step)
            self.r_x_ij += Plam * tmp_1d

            tmp_2d *=0
            functionsC.grad_field_ij(tmp_2d,lam,self.center,self.space_discretization_step)
            self.grad_r_x_ij  += Plam * tmp_2d

            tmp_2d *=0
            functionsC.grad_laplace_field_ij(tmp_2d,lam,self.center,self.space_discretization_step)
            self.grad_Laplace_r_x_ij += Plam * tmp_2d

            tmp_1d *=0
            tmp_4d *=0
            functionsC.hess_field_ij(tmp_4d,tmp_1d,lam,self.center,self.space_discretization_step)
            self.Laplace_r_x_ij += Plam * tmp_1d
            H_ij += Plam * tmp_4d
        
        self.H_r_x_ij = H_ij.reshape(self.N,self.N,2,2)
        
        # TODO: maybe we can combine certain fields into one (e.g., r_finite = total rate averaged over circumference)
        # TODO: maybe use linearized update equation by Ben [Eq. (3) in version from 08.10.2021]

        self.log_r_x_ij = np.log(self.r_x_ij)
        # total binding rate (averaged over circumference of agent) for all possible positions of source
        self.rfinite_x_ij = self.r_x_ij + (self.agent_size/2)**2*(self.H_r_x_ij[:,:,0,0]+self.H_r_x_ij[:,:,1,1])
        self.log_rfinite_x_ij = np.log(self.rfinite_x_ij)
        self.grad_rfinite_x_ij = self.grad_r_x_ij + (self.agent_size/2)**2*self.grad_Laplace_r_x_ij
