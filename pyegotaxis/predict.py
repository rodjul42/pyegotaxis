import numpy as np
import math
import pandas as pd
import os

from .simdata import simdata
from . import _cpredict
from .functions import *
 
import logging
logger = logging.getLogger(__name__)


class predict_base():
    def __init__(self,sim_data):
        self.orientation = 0
        return 

    def check_stability(self,sim_data):
        """
        Check if all stavility criteria are furfilled.
        """
        # NOTE: N << l*np.sqrt(8/(D*dt)) [Courant criterion: for numerical stability of Euler scheme for diffusion]
        D_stability = (4*sim_data.D*sim_data.dt) / sim_data.space_discretization_step**2
        if D_stability  > 0.1:
            logger.warning(f'Stability for diffuision is only {D_stability} ')
        elif D_stability > 1:
            logger.critical(f'unstable for diffuision: {D_stability} ')
            raise ValueError('unstable for diffuision')
        # NOTE: N << 4*l/(speed*dt)  [Stability criterion for convection]
        C_stability = sim_data.speed*sim_data.dt / sim_data.space_discretization_step
        if C_stability > 0.1:
            logger.warning(f'Stability for convection is only {C_stability} ')
        elif C_stability > 1:
            logger.critical(f'unstable for convection: {C_stability} ')
            raise ValueError('unstable for convection')

        #stability rotation diffusion

        return

    def log(self,update_step):
        return None

    def process_log(self,data_t):    
        if data_t[0] is None:
            return None
        else:
            return np.array(data_t)

    def update_likelyhood(self,L,dx,dt):
        """
        Prediction step, accounting for convection (due to active motion of the agent) and diffusion (due to motility noise of the agent)

        """
        return L.copy()


class predict_adv_diff_diffrot(predict_base):
    def __init__(self,sim_data,orientation=0,skipL=False):
        '''
        Calulate advection rotatinal diffusion, translational diffusion not yet
        warning does not work with adaptive time step
        sim_data     : simdata class instance
        orientation  : initial orientation [rad]
        skipL        : so not update the likelihood
        '''
        super().__init__(sim_data)
        self.space_discretization_step = sim_data.space_discretization_step
        self.D = sim_data.D
        if self.D != 0:
            raise NotImplementedError('')
        self.Drot = sim_data.Drot
        self.DrotReal = sim_data.DrotReal
        self.center = sim_data.center
        self.orientation = orientation

        self.repeat = 1
        self.check_stability(sim_data)
        if self.repeat>1:
            logger.info(f'repeat  {self.repeat} ')
        self.dt_rot = sim_data.dt/self.repeat
        if self.repeat == 1:
            self.update_likelyhood = self.update_likelyhood1
        elif self.repeat == 2:
            self.update_likelyhood = self.update_likelyhood2
        elif self.repeat == 3:
            self.update_likelyhood = self.update_likelyhood3
        elif self.repeat == 4:
            self.update_likelyhood = self.update_likelyhood4
        else:
            if self.repeat%2:
                self.update_likelyhood = self.update_likelyhoodn_odd
            else:
                self.update_likelyhood = self.update_likelyhoodn_even

        if skipL:
            self.update_likelyhood = super().update_likelyhood
            logger.warning(f'skip calulaton of likelyhood ')
        return 

    def check_stability(self,sim_data):
        """
        Check if all stavility criteria are furfilled.
        """
        # NOTE: N << l*np.sqrt(8/(D*dt)) [Courant criterion: for numerical stability of Euler scheme for diffusion]
        Dmax = sim_data.Drot*(sim_data.center*sim_data.center)
        
        while True:
            DS = False
            AS = False
            D_stability = (4*Dmax*sim_data.dt/self.repeat) / sim_data.space_discretization_step**2
            if D_stability  > 0.4:
                #logger.warning(f'Stability for diffuision is only {D_stability} ')
                pass
            elif D_stability > 1:
                #logger.critical(f'unstable for diffuision: {D_stability} ')
                #raise ValueError('unstable for diffuision')
                pass
            else:
                DS = True
            # NOTE: N << 4*l/(speed*dt)  [Stability criterion for convection]
            speed = max(sim_data.Drot*(sim_data.center),sim_data.speed)
            C_stability = speed*sim_data.dt/self.repeat / sim_data.space_discretization_step
            if C_stability > 0.1:
                pass
                #logger.warning(f'Stability for convection is only {C_stability} ')
            elif C_stability > 1:
                pass
                #logger.critical(f'unstable for convection: {C_stability} ')
                #raise ValueError('unstable for convection')
            else:
                AS = True
            if AS and DS:
                break
            self.repeat += 1
        return D_stability,C_stability


    def log(self,update_step):
        return self.orientation

    def update_source(self,source,dx,dt):
        """
        Translational diffusion of agent > translational diffusion of source in egocentric map
        Rotatinal diffusion of agent > rotate source
        source      : position of source
        dx          : displacement vector of agent
        dt          : time step
        """
        phi = np.random.standard_normal(1)[0] * math.sqrt(2*self.DrotReal*dt)
        #phi = 2*np.pi/40*dt
        self.orientation -= phi
        d_Drot =  np.array([[math.cos(phi),-math.sin(phi)],
                            [math.sin(phi),math.cos(phi)]])
        
        return d_Drot@source - dx*dt #+ np.random.standard_normal(2) * math.sqrt(2*self.D*dt)

    def update_likelyhood1(self,L,dx,dt):
        """
        Prediction step, accounting for convection (due to active motion of the agent) and diffusion (due to motility noise of the agent)
        """
        res = np.zeros_like(L)
        _cpredict.predict_all(L,res,self.space_discretization_step,dx[0]*self.dt_rot,dx[1]*self.dt_rot,self.D,self.Drot,self.center,self.center,self.dt_rot) 
        return res

    def update_likelyhood2(self,L,dx,dt):
        """
        Prediction step, accounting for convection (due to active motion of the agent) and diffusion (due to motility noise of the agent)
        """
        res = np.zeros_like(L)      
        _cpredict.predict_all(L,res,self.space_discretization_step,dx[0]*self.dt_rot,dx[1]*self.dt_rot,self.D,self.Drot,self.center,self.center,self.dt_rot) 
        _cpredict.predict_all(res,L,self.space_discretization_step,dx[0]*self.dt_rot,dx[1]*self.dt_rot,self.D,self.Drot,self.center,self.center,self.dt_rot) 
        return L 

    def update_likelyhood3(self,L,dx,dt):
        """
        Prediction step, accounting for convection (due to active motion of the agent) and diffusion (due to motility noise of the agent)
        """
        
        res = np.zeros_like(L)
        _cpredict.predict_all(L,res,self.space_discretization_step,dx[0]*self.dt_rot,dx[1]*self.dt_rot,self.D,self.Drot,self.center,self.center,self.dt_rot) 
        _cpredict.predict_all(res,L,self.space_discretization_step,dx[0]*self.dt_rot,dx[1]*self.dt_rot,self.D,self.Drot,self.center,self.center,self.dt_rot)
        _cpredict.predict_all(L,res,self.space_discretization_step,dx[0]*self.dt_rot,dx[1]*self.dt_rot,self.D,self.Drot,self.center,self.center,self.dt_rot) 
        return res 

    def update_likelyhood4(self,L,dx,dt):
        """
        Prediction step, accounting for convection (due to active motion of the agent) and diffusion (due to motility noise of the agent)
        """

        res = np.zeros_like(L)
        _cpredict.predict_all(L,res,self.space_discretization_step,dx[0]*self.dt_rot,dx[1]*self.dt_rot,self.D,self.Drot,self.center,self.center,self.dt_rot) 
        _cpredict.predict_all(res,L,self.space_discretization_step,dx[0]*self.dt_rot,dx[1]*self.dt_rot,self.D,self.Drot,self.center,self.center,self.dt_rot) 
        _cpredict.predict_all(L,res,self.space_discretization_step,dx[0]*self.dt_rot,dx[1]*self.dt_rot,self.D,self.Drot,self.center,self.center,self.dt_rot) 
        _cpredict.predict_all(res,L,self.space_discretization_step,dx[0]*self.dt_rot,dx[1]*self.dt_rot,self.D,self.Drot,self.center,self.center,self.dt_rot) 
        return L 

    def update_likelyhoodn_even(self,L,dx,dt):
        """
        Prediction step, accounting for convection (due to active motion of the agent) and diffusion (due to motility noise of the agent)
        """

        res = np.zeros_like(L)
        for i in range(int(self.repeat/2)):
            _cpredict.predict_all(L,res,self.space_discretization_step,dx[0]*self.dt_rot,dx[1]*self.dt_rot,self.D,self.Drot,self.center,self.center,self.dt_rot) 
            _cpredict.predict_all(res,L,self.space_discretization_step,dx[0]*self.dt_rot,dx[1]*self.dt_rot,self.D,self.Drot,self.center,self.center,self.dt_rot)             
        return L 

    def update_likelyhoodn_odd(self,L,dx,dt):
        """
        Prediction step, accounting for convection (due to active motion of the agent) and diffusion (due to motility noise of the agent)
        """

        res = np.zeros_like(L)
        for i in range(int(self.repeat/2)):
            _cpredict.predict_all(L,res,self.space_discretization_step,dx[0]*self.dt_rot,dx[1]*self.dt_rot,self.D,self.Drot,self.center,self.center,self.dt_rot) 
            _cpredict.predict_all(res,L,self.space_discretization_step,dx[0]*self.dt_rot,dx[1]*self.dt_rot,self.D,self.Drot,self.center,self.center,self.dt_rot)             
        _cpredict.predict_all(L,res,self.space_discretization_step,dx[0]*self.dt_rot,dx[1]*self.dt_rot,self.D,self.Drot,self.center,self.center,self.dt_rot) 
        return res 


class predict_adv_diff_diffrot_Adrift(predict_adv_diff_diffrot):
    def __init__(self,sim_data,vdrift=0,orientation=0,skipL=False):
        '''
        Here the added drift from the agent is impemented.
        Calulate advection rotatinal diffusion, translational diffusion not yet
        warning does not work with adaptive time step
        sim_data     : simdata class instance
        orientation  : initial orientation [rad]
        skipL        : so not update the likelihood
        '''
        super().__init__(sim_data,orientation=orientation,skipL=skipL)
        self.vdrift=vdrift

    def update_source(self,source,dx,dt):
        """
        Translational diffusion of agent > translational diffusion of source in egocentric map
        Rotatinal diffusion of agent > rotate source
        source      : position of source
        dx          : displacement vector of agent
        dt          : time step
        """
        phi = np.random.standard_normal(1)[0] * math.sqrt(2*self.DrotReal*dt)
        self.orientation -= phi
        d_Drot =  np.array([[math.cos(phi),-math.sin(phi)],
                            [math.sin(phi),math.cos(phi)]])
 
        return d_Drot@source - dx*dt - np.array([math.cos(self.orientation),-math.sin(self.orientation)])*self.vdrift*dt
class predict_diffrot(predict_base):
    def __init__(self,sim_data,orientation=0,skipL=False):
        '''
        warning does not work with adaptive time step
        orientation  : initial orientation [rad]
        '''
        super().__init__(sim_data)
        self.space_discretization_step = sim_data.space_discretization_step
        self.D = sim_data.D
        self.Drot = sim_data.Drot
        self.DrotReal = sim_data.DrotReal
        self.center = sim_data.center
        self.orientation = orientation

        self.repeat = 1
        self.check_stability(sim_data)
        if self.repeat>1:
            logger.info(f'repeat  {self.repeat} ')
        self.dt_rot = sim_data.dt/self.repeat
        if self.repeat == 1:
            self.update_likelyhood = self.update_likelyhood1
        elif self.repeat == 2:
            self.update_likelyhood = self.update_likelyhood2
        elif self.repeat == 3:
            self.update_likelyhood = self.update_likelyhood3
        elif self.repeat == 4:
            self.update_likelyhood = self.update_likelyhood4
        else:
            if self.repeat%2:
                self.update_likelyhood = self.update_likelyhoodn_odd
            else:
                self.update_likelyhood = self.update_likelyhoodn_even

        if skipL:
            self.update_likelyhood = super().update_likelyhood
            logger.warning(f'skip calulaton of likelyhood ')
        return 

    def check_stability(self,sim_data):
        """
        Check if all stavility criteria are furfilled.
        """
        # NOTE: N << l*np.sqrt(8/(D*dt)) [Courant criterion: for numerical stability of Euler scheme for diffusion]
        Dmax = sim_data.Drot*(sim_data.center*sim_data.center)
        
        while True:
            DS = False
            AS = False
            D_stability = (4*Dmax*sim_data.dt/self.repeat) / sim_data.space_discretization_step**2
            if D_stability  > 0.4:
                #logger.warning(f'Stability for diffuision is only {D_stability} ')
                pass
            elif D_stability > 1:
                #logger.critical(f'unstable for diffuision: {D_stability} ')
                #raise ValueError('unstable for diffuision')
                pass
            else:
                DS = True
            # NOTE: N << 4*l/(speed*dt)  [Stability criterion for convection]
            speed = max(sim_data.Drot*(sim_data.center),sim_data.speed)
            C_stability = speed*sim_data.dt/self.repeat / sim_data.space_discretization_step
            if C_stability > 0.1:
                pass
                #logger.warning(f'Stability for convection is only {C_stability} ')
            elif C_stability > 1:
                pass
                #logger.critical(f'unstable for convection: {C_stability} ')
                #raise ValueError('unstable for convection')
            else:
                AS = True
            if AS and DS:
                break
            self.repeat += 1
        return D_stability,C_stability
    def log(self,update_step):
        return self.orientation

    def update_source(self,source,dx,dt):
        """
        Translational diffusion of agent > translational diffusion of source in egocentric map
        Rotatinal diffusion of agent > rotate source
        dx          : displacement vector of agent
        dt          : time step
        """
        phi = np.random.standard_normal(1)[0] * math.sqrt(2*self.DrotReal*dt)
        self.orientation -= phi
        d_Drot =  np.array([[math.cos(phi),-math.sin(phi)],
                            [math.sin(phi),math.cos(phi)]])
        
        return d_Drot@source 

    def update_likelyhood1(self,L,dx,dt):
        """
        Prediction step, accounting for convection (due to active motion of the agent) and diffusion (due to motility noise of the agent)
        """
        dx *= 0
        res = np.zeros_like(L)
        _cpredict.predict_rot(L,res,self.space_discretization_step,self.Drot,self.center,self.center,self.dt_rot) 
        return res

    def update_likelyhood2(self,L,dx,dt):
        """
        Prediction step, accounting for convection (due to active motion of the agent) and diffusion (due to motility noise of the agent)
        """
        dx *= 0
        res = np.zeros_like(L)      
        _cpredict.predict_rot(L,res,self.space_discretization_step,self.Drot,self.center,self.center,self.dt_rot) 
        _cpredict.predict_rot(res,L,self.space_discretization_step,self.Drot,self.center,self.center,self.dt_rot) 
        return L 

    def update_likelyhood3(self,L,dx,dt):
        """
        Prediction step, accounting for convection (due to active motion of the agent) and diffusion (due to motility noise of the agent)
        """
        dx *= 0
        res = np.zeros_like(L)
        _cpredict.predict_rot(L,res,self.space_discretization_step,self.Drot,self.center,self.center,self.dt_rot) 
        _cpredict.predict_rot(res,L,self.space_discretization_step,self.Drot,self.center,self.center,self.dt_rot)
        _cpredict.predict_rot(L,res,self.space_discretization_step,self.Drot,self.center,self.center,self.dt_rot) 
        return res 

    def update_likelyhood4(self,L,dx,dt):
        """
        Prediction step, accounting for convection (due to active motion of the agent) and diffusion (due to motility noise of the agent)
        """
        dx *= 0
        res = np.zeros_like(L)
        _cpredict.predict_rot(L,res,self.space_discretization_step,self.Drot,self.center,self.center,self.dt_rot) 
        _cpredict.predict_rot(res,L,self.space_discretization_step,self.Drot,self.center,self.center,self.dt_rot) 
        _cpredict.predict_rot(L,res,self.space_discretization_step,self.Drot,self.center,self.center,self.dt_rot) 
        _cpredict.predict_rot(res,L,self.space_discretization_step,self.Drot,self.center,self.center,self.dt_rot) 
        return L 

    def update_likelyhoodn_even(self,L,dx,dt):
        """
        Prediction step, accounting for convection (due to active motion of the agent) and diffusion (due to motility noise of the agent)
        """
        dx *= 0
        res = np.zeros_like(L)
        for i in range(int(self.repeat/2)):
            _cpredict.predict_rot(L,res,self.space_discretization_step,self.Drot,self.center,self.center,self.dt_rot) 
            _cpredict.predict_rot(res,L,self.space_discretization_step,self.Drot,self.center,self.center,self.dt_rot)             
        return L 

    def update_likelyhoodn_odd(self,L,dx,dt):
        """
        Prediction step, accounting for convection (due to active motion of the agent) and diffusion (due to motility noise of the agent)
        """
        dx *= 0
        res = np.zeros_like(L)
        for i in range(int(self.repeat/2)):
            _cpredict.predict_rot(L,res,self.space_discretization_step,self.Drot,self.center,self.center,self.dt_rot) 
            _cpredict.predict_rot(res,L,self.space_discretization_step,self.Drot,self.center,self.center,self.dt_rot)             
        _cpredict.predict_rot(L,res,self.space_discretization_step,self.Drot,self.center,self.center,self.dt_rot) 
        return res 




from multiprocessing import Pool
def calc_i(i,sim_data):
    """
    Calulate the kernel for peak at index i of likelihood (flatten)
    i : index of delta peak
    sim_data     : simdata class instance
    """
    DROT = predict_diffrot(sim_data)
    I0 = np.zeros_like(sim_data.Linit)
    I0.ravel()[i] = 1
    I1 = DROT.update_likelyhood(I0,[0,0],sim_data.dt).ravel()
    # realative threshold for the kernel. 1e-4 with Drot=1 perfect for 70 time units.
    # with Drot=0.1 it is less good , but depends on the seed.....
    # so 1e-4 seems like a good value -> For gaus cutoff at X=+-~4.3 sigma
    th = np.max(I1)*1e-4
    row = np.where(np.logical_or(I1>th,I1<-th))[0]
    return i,row,I1[row]


def calc_kernel(Drot,dt=5e-3,N=200,l=1,threads = 4,kernel_path = "/home/jrode/kernels"):
    """
    Calulate kernel for predict_adv_diffrot_kernel
    Drot : rotatinal diffusion constant [radian^2/s]
    dt   : time step [s]
    N    : Number of elements along one axis of the likelihood
    l    : half lenght of one axis of the likelihood [m]
    threads: threads used for calulation
    kernel_path : output path
    """
    sim_data = simdata(maxTime=1000,dt=dt,l=l,N=N,D=0.00,Drot=Drot)
    
    kernel_name = f"kernel_D{sim_data.Drot}_dt{sim_data.dt}_N{sim_data.N}_l{sim_data.l}"
    path_kernel_val = os.path.join(kernel_path, kernel_name+'_val.npz')
    path_kernel_idx = os.path.join(kernel_path, kernel_name+'_idx.npz')
    if os.path.isfile(path_kernel_val) and os.path.isfile(path_kernel_idx):
        logger.info(f'kernel {path_kernel_val} and {path_kernel_idx} already exist')
        return
    print('kernels not found',path_kernel_val,path_kernel_idx)
    with Pool(threads) as p:
        kernels_raw = p.starmap(calc_i,[(i,sim_data) for i in range(np.prod(sim_data.Linit.shape))])
    np.savez_compressed(path_kernel_idx,**{str(i) : idx for i,idx,val in kernels_raw} )
    np.savez_compressed(path_kernel_val,**{str(i) : val for i,idx,val in kernels_raw} )

class predict_adv_diffrot_kernel(predict_base):
    def __init__(self,sim_data,orientation=0,kernel_path='.'):
        '''
        Calulate advection rotatinal diffusion using a larger kernel.
        Faster for large Drot
        warning does not work with adaptive time step
        sim_data     : simdata class instance
        orientation  : initial orientation [rad]
        kernel_path  : path for the kernels ()
        '''
        super().__init__(sim_data)
        self.space_discretization_step = sim_data.space_discretization_step
        self.Drot = sim_data.Drot
        self.DrotReal = sim_data.DrotReal
        self.dt = sim_data.dt
        self.orientation = orientation
        
        kernel_name = f"kernel_D{self.Drot}_dt{self.dt}_N{sim_data.N}_l{sim_data.l}"
        path_kernel_val = os.path.join(kernel_path, kernel_name+'_val.npz')
        path_kernel_idx = os.path.join(kernel_path, kernel_name+'_idx.npz')
        if not (os.path.isfile(path_kernel_val) or os.path.isfile(path_kernel_idx)):
            raise FileExistsError(f'Kernels does not exist {path_kernel_idx}')
        self.cDrot = _cpredict.new_calc_Drot(sim_data.minL)
        A = _cpredict.calc_Drot_load_kernel_val(self.cDrot,path_kernel_val)
        B = _cpredict.calc_Drot_load_kernel_idx(self.cDrot,path_kernel_idx)
        assert(A==B)

        self.check_stability(sim_data)
        return
    
    def __del__(self):
        _cpredict.delete_calc_Drot(self.cDrot)

    def check_stability(self,sim_data):
        """
        Check if all stavility criteria are furfilled.
        """
        # NOTE: N << l*np.sqrt(8/(D*dt)) [Courant criterion: for numerical stability of Euler scheme for diffusion]
        C_stability = sim_data.speed*sim_data.dt / sim_data.space_discretization_step
        if C_stability > 1:
            logger.critical(f'unstable for convection: {C_stability} ')
            raise ValueError('unstable for convection')
        elif C_stability > 0.1:
            logger.warning(f'Stability for convection is only {C_stability} ')
        else:
            pass
        

    def log(self,update_step):
        return self.orientation

    def update_source(self,source,dx,dt):
        """
        Translational diffusion of agent > translational diffusion of source in egocentric map
        Rotatinal diffusion of agent > rotate source
        dx          : displacement vector of agent
        dt          : time step
        """
        phi = np.random.standard_normal(1)[0] * math.sqrt(2*self.DrotReal*dt)
        self.orientation -= phi
        d_Drot =  np.array([[math.cos(phi),-math.sin(phi)],
                            [math.sin(phi),math.cos(phi)]])
        
        return d_Drot@source - dx*dt 

    def update_likelyhood(self,L,dx,dt):
        """
        Prediction step, accounting for convection (due to active motion of the agent) and diffusion (due to motility noise of the agent)
        """
        res = np.zeros_like(L)
        _cpredict.calc_Drot_predict(self.cDrot,L,res)
        _cpredict.predict_adv(res,L,dx[0]*dt,dx[1]*dt, self.space_discretization_step,dt) 
        return L


class predict_adv_diffrot_kernel_INFO_delta(predict_adv_diffrot_kernel):
    index = ['S_rot','Sphi_rot','Sr_rot','S_adv','Sphi_adv','Sr_adv']
    def __init__(self,sim_data,orientation=0,kernel_path='.'):
        '''
        Tracks the change of entropy for each step and logs the summed changes of sampling_interval steps
        Calulate advection rotatinal diffusion using a larger kernel.
        warning does not work with adaptive time step
        sim_data     : simdata class instance
        orientation  : initial orientation [rad]
        kernel_path  : path for the kernels ()
        '''
        super().__init__(sim_data=sim_data,orientation=orientation,kernel_path=kernel_path)
        self.Es = np.zeros((sim_data.sampling_interval,len(self.index)))
        self.Entropy_last_step = 0
        self.i = 0
        self.sim_data = sim_data

        return
    

    def log(self,update_stepES):
        ###### save the sum entropy change and orientation
        results = np.zeros((self.sim_data.sampling_interval,len(self.index)))
        results[0,:3] = self.Es[0,:3] - self.Entropy_last_step #delta from old entropies to after drot ; for first use saved in slot0
        results[1:,:3] = self.Es[1:,:3] - update_stepES[:-1,:3] #delta from old entropies after update to after drot 

        results[:,3:] = self.Es[:,3:] - self.Es[:,:3]  # delta from after drot to after advection
        self.i = 0
        self.Entropy_last_step = np.copy(update_stepES[self.sim_data.sampling_interval-1,:3])
        self.Es *= 0
        return self.orientation,np.sum(results,axis=0)

    def process_log(self,data_t):    
        if data_t[0] is None:
            return None
        else:
            entropies = np.zeros((len(data_t),len(self.index)))
            orientations = np.zeros(len(data_t))
            for i,(orientation,result) in enumerate(data_t):
                entropies[i] = result
                orientations[i] = orientation
            return np.array(orientations),pd.DataFrame(entropies,columns=self.index)

    def update_likelyhood(self,L,dx,dt):
        """
        Prediction step, accounting for convection (due to active motion of the agent) and diffusion (due to motility noise of the agent)
        """

        res = np.zeros_like(L)
        _cpredict.calc_Drot_predict(self.cDrot,L,res)
        _cpredict.predict_adv(res,L,dx[0]*dt,dx[1]*dt, self.space_discretization_step,dt) 


        L1 = np.maximum( res, self.sim_data.minL )
        L1[self.sim_data.nogo_area] = self.sim_data.minL
        L1 /= (np.sum(L1))

        L2 = np.maximum( L, self.sim_data.minL )
        L2[self.sim_data.nogo_area] = self.sim_data.minL
        L2 /= (np.sum(L2))
        
        ###### save entropy changes
        self.Es[self.i] = [entropy(L1),
                   entropy_phi(L1,self.sim_data),
                   entropy_r(L1,self.sim_data),
                   entropy(L2),
                   entropy_phi(L2,self.sim_data),
                   entropy_r(L2,self.sim_data)]
        self.i += 1
        return L





class predict_adv_INFO_delta(predict_base):
    index = ['S_rot','Sphi_rot','Sr_rot','S_adv','Sphi_adv','Sr_adv']
    def __init__(self,sim_data,orientation=0):
        '''
        Calulate advection rotatinal diffusion using a larger kernel.
        Faster for large Drot
        warning does not work with adaptive time step
        sim_data     : simdata class instance
        orientation  : initial orientation [rad]
        kernel_path  : path for the kernels ()
        '''
        super().__init__(sim_data)
        self.space_discretization_step = sim_data.space_discretization_step
        self.Drot = sim_data.Drot
        self.DrotReal = sim_data.DrotReal
        self.dt = sim_data.dt
        self.orientation = orientation
        self.check_stability(sim_data)
        self.Es = np.zeros((sim_data.sampling_interval,len(self.index)))
        self.Entropy_last_step = 0
        self.i = 0
        self.sim_data = sim_data
        self.repeat = 1
        self.dt2 = sim_data.dt/self.repeat
        return
    

    def log(self,update_stepES):
        ###### save the sum entropy change and orientation
        results = np.zeros((self.sim_data.sampling_interval,len(self.index)))
        results[0,:3] = self.Es[0,:3] - self.Entropy_last_step #delta from old entropies to after drot ; for first use saved in slot0
        results[1:,:3] = self.Es[1:,:3] - update_stepES[:-1,:3] #delta from old entropies after update to after drot 

        results[:,3:] = self.Es[:,3:] - self.Es[:,:3]  # delta from after drot to after advection
        self.i = 0
        self.Entropy_last_step = np.copy(update_stepES[self.sim_data.sampling_interval-1,:3])
        self.Es *= 0
        return self.orientation,np.sum(results,axis=0)

    def process_log(self,data_t):    
        if data_t[0] is None:
            return None
        else:
            entropies = np.zeros((len(data_t),len(self.index)))
            orientations = np.zeros(len(data_t))
            for i,(orientation,result) in enumerate(data_t):
                entropies[i] = result
                orientations[i] = orientation
            return np.array(orientations),pd.DataFrame(entropies,columns=self.index)



    

    def check_stability(self,sim_data):
        """
        Check if all stavility criteria are furfilled.
        """
        # NOTE: N << l*np.sqrt(8/(D*dt)) [Courant criterion: for numerical stability of Euler scheme for diffusion]
        C_stability = sim_data.speed*sim_data.dt / sim_data.space_discretization_step
        if C_stability > 1:
            logger.critical(f'unstable for convection: {C_stability} ')
            raise ValueError('unstable for convection')
        elif C_stability > 0.1:
            logger.warning(f'Stability for convection is only {C_stability} ')
        else:
            pass
        


    def update_source(self,source,dx,dt):
        """
        Translational diffusion of agent > translational diffusion of source in egocentric map
        Rotatinal diffusion of agent > rotate source
        dx          : displacement vector of agent
        dt          : time step
        """
        return source - dx*dt 

    def update_likelyhood(self,L,dx,dt):
        """
        Prediction step, accounting for convection (due to active motion of the agent) and diffusion (due to motility noise of the agent)
        """
        L1 = np.maximum( L, self.sim_data.minL )
        L1[self.sim_data.nogo_area] = self.sim_data.minL
        L1 /= (np.sum(L1))
        
        res = np.zeros_like(L)
        for i in range(self.repeat):
            _cpredict.predict_adv(L,res,dx[0]*self.dt2,dx[1]*self.dt2, self.space_discretization_step,self.dt2) 
            L = res.copy()
            
  

        L2 = np.maximum( L, self.sim_data.minL )
        L2[self.sim_data.nogo_area] = self.sim_data.minL
        L2 /= (np.sum(L2))
        
        ###### save entropy changes
        self.Es[self.i] = [entropy(L1),
                   entropy_phi(L1,self.sim_data),
                   entropy_r(L1,self.sim_data),
                   entropy(L2),
                   entropy_phi(L2,self.sim_data),
                   entropy_r(L2,self.sim_data)]
        self.i += 1
        return L