import numpy as np
import numpy.linalg as algebra
    
from .functions import *
from .utils import *
from copy import deepcopy
import pandas as pd
class update_base:
    def __init__(self) -> None:
        pass

    def update(self,norm_dm,dm,L_predict,dt,sim_data):
        return L_predict.copy()

    def log(self,predict):
        return None


    def process_log(self,update_t):    
        if update_t[0] is None:
            return None
        else:
            return np.array(update_t)
class ben(update_base):
    def update(self,norm_dm,dm,L_predict,dt,sim_data):
        """
        linearized update step from Eq. (4) in version 08.10.2021 
        equation are from our paper where grad is to x_0 but grad is still defined with respect to x
        Hence dm need a negative sign
        """

        # Compute various mean values <.> with respect to the current likelihood map; need without event
        rfinite_mean = np.sum(np.multiply(L_predict,sim_data.rfinite_x_ij))
        # ... terms present even in absence of an event
        L = L_predict + L_predict*(rfinite_mean-sim_data.rfinite_x_ij)*dt 
        # ... additional terms if event occured [linearized update equation by Ben, Eq. (4)]
        if (norm_dm>0):
            
            dm = dm*-1
            # Compute various mean values <.> with respect to the current likelihood map; need with event
            r_mean = np.sum(np.multiply(L_predict,sim_data.r_x_ij))
            mean_grad_r = np.array([np.sum(np.multiply(L_predict,sim_data.grad_r_x_ij[:,:,0])),
                                    np.sum(np.multiply(L_predict,sim_data.grad_r_x_ij[:,:,1]))])
            #Laplace_mean = np.sum(np.multiply(L_predict,sim_data.Laplace_r_x_ij))
            
            L = L + L_predict*( \
                norm_dm*(sim_data.rfinite_x_ij/rfinite_mean-1) + \
                sim_data.agent_size * np.dot( sim_data.grad_r_x_ij, dm - norm_dm*0.5*sim_data.agent_size*mean_grad_r/r_mean )/r_mean - \
                sim_data.agent_size * np.dot( mean_grad_r,          dm - norm_dm*0.5*sim_data.agent_size*mean_grad_r/r_mean )*sim_data.r_x_ij/(r_mean*r_mean) \
                )
        return L

class ben_sc(update_base):
    def update(self,norm_dm,dm,L_predict,dt,sim_data):
        """
        linearized update step from Eq. (4) but only using gradient sensing term
        """
                
        # ... additional terms if event occured [linearized update equation by Ben, Eq. (3) in version 08.10.2021]
        if (norm_dm>0):
            #if (norm_dm>1):
            #    print('Warning, ||dm|| =',norm_dm)
            #    # TODO: approximate update rule for case of >1 event per time step
            rfinite_mean = np.sum(np.multiply(L_predict,sim_data.rfinite_x_ij))
            dm = dm*-1
            # Compute various mean values <.> with respect to the current likelihood map; need with event
            r_mean = np.sum(np.multiply(L_predict,sim_data.r_x_ij))
            mean_grad_r = np.array([np.sum(np.multiply(L_predict,sim_data.grad_r_x_ij[:,:,0])),
                                    np.sum(np.multiply(L_predict,sim_data.grad_r_x_ij[:,:,1]))])
            #Laplace_mean = np.sum(np.multiply(L_predict,sim_data.Laplace_r_x_ij))
            A = dm - norm_dm*0.5*sim_data.agent_size*mean_grad_r/rfinite_mean
            
            L = L_predict + L_predict*sim_data.agent_size*(np.dot(sim_data.grad_r_x_ij,A) / sim_data.r_x_ij  - np.dot(mean_grad_r/r_mean,A))*sim_data.r_x_ij/r_mean 
            #L *= (sim_data.R_x_ij>0.1)
            #L /= L.sum()
            return L 
        else:
            return L_predict.copy()


class ben_INFO_delta(ben):
    ''' Save all directional information changes
    '''
    index = ['S_t','Sphi_t','Sr_t', 'S_sc','Sphi_sc','Sr_sc', 'S_tc','Sphi_tc','Sr_tc']

    def __init__(self,sim_data) -> None:
        super().__init__()

        self.sim_data_tc = deepcopy(sim_data)
        self.sim_data_tc.agent_size = 0
        self.sim_data_tc.calc_matrices()
        self.Es = np.zeros((sim_data.sampling_interval,len(self.index)))
        self.i = 0

    def update(self,norm_dm,dm,L_predict,dt,sim_data):

        ########### Update L only sc 
        u = ben_sc()
        Lscn = u.update(norm_dm,dm,L_predict,dt,sim_data)
        Lscn = np.maximum( Lscn, sim_data.minL )
        Lscn[sim_data.nogo_area] = sim_data.minL
        Lscn /= (np.sum(Lscn))
        
        
        ######## calc for temporal comparison
        ########### update with a=0
        Ltc = super().update(norm_dm,dm,L_predict,dt,self.sim_data_tc)
        Ltc = np.maximum( Ltc, sim_data.minL )
        Ltc[sim_data.nogo_area] = sim_data.minL
        Ltc /= (np.sum(Ltc))

        ########### normal update
        L = super().update(norm_dm,dm,L_predict,dt,sim_data)

        ########### calc entropy change normal update
        L1 = np.maximum( L, sim_data.minL )
        L1[sim_data.nogo_area] = sim_data.minL
        L1 /= (np.sum(L1))

        
        ###### save entropy changes
        self.Es[self.i] = [ entropy( L1),
                    entropy_phi(L1,sim_data),
                    entropy_r(L1,sim_data),
                    entropy( Lscn),
                    entropy_phi(Lscn,sim_data),
                    entropy_r(Lscn,sim_data),
                    entropy( Ltc) ,
                    entropy_phi(Ltc,sim_data),
                    entropy_r(Ltc,sim_data)       ]
        self.i += 1

        return L

    def log(self,predictEs):
        ###### save the sum entropy change
        results = np.zeros_like(self.Es)
        results[:,:3] = self.Es[:,:3] - predictEs[:,3:] # delta after adv (predict.Es[1:,3:]) to after total update
        results[:,3:6] = self.Es[:,3:6] - predictEs[:,3:] # delta after adv (predict.Es[1:,3:]) to after sc update
        results[:,6:] = self.Es[:,6:] - predictEs[:,3:] # delta after adv (predict.Es[1:,3:]) to after tc update
        self.i = 0
        self.Es *= 0
        return np.sum(results,axis=0)
        

    def process_log(self,update_t):    
        if update_t[0] is None:
            return None
        else:
            return pd.DataFrame(update_t,columns=self.index)