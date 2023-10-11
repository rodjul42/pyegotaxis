## 'decision.py' = library implementing decision strategy, called by 'main.py' for Bayesian chemotaxis
# Andra Auconi, Benjamin Friedrich, 2020-2021

# import standard libraries
import numpy as np
import numpy.linalg as algebra
import math
from .functions import xy_ij


class decision_base():
    '''
    decision base class
    '''
    def __init__(self,sim_data):
        self.sim_data = sim_data
        self.calc_d_matrices()
    
    def calc_d_matrices(self):
        return

    def log(self):
        return None

    
class stubborn(decision_base):
    '''
    agent just goes in one direction
    '''
    def __init__(self,sim_data,phi=0):
        """
        sim_data : sim_data instance with the parameters and precomputed matrices
        """
        self.direction = np.array([np.cos(phi),np.sin(phi)])
        
    def decision(self,L,source):
        return self.direction

class godlike(decision_base):
    '''
    agent has perfect knowlage of source and moves towrds it
    '''
    def decision(self,L,source):
        return source.copy()/(math.sqrt(source[0]*source[0]+source[1]*source[1]))

class drunk(decision_base):
    '''
    agent has no idea and does random walk
    '''
    def decision(self,L,source):
        phi = np.random.random_sample()*2*np.pi
        return np.array([np.cos(phi),np.sin(phi)])

class Infotaxis_ben(decision_base):
    """
    Infotaxis
    decision routine from Eq. (6) 
    equation are from our paper where grad is to x_0 but grad is still defined with respect to x
    since the terms have an odd number of gradients there is a change of signs by the defintion of the 
    move_direction : move_direction_x = - t0_x-t1_x
    """
        
    def calc_d_matrices(self):
        # second term second part of product rule of Eq. (5) in version 08.10.2021] note ||**2 is multiply out
        # grad|grad c / c|**2
        self.eq5_term1_1 =  np.array([ 2/(self.sim_data.r_x_ij*self.sim_data.r_x_ij)*( \
                        self.sim_data.grad_r_x_ij[:,:,0]*(self.sim_data.H_r_x_ij[:,:,i,0]-self.sim_data.grad_r_x_ij[:,:,0]*self.sim_data.grad_r_x_ij[:,:,i]/self.sim_data.r_x_ij) \
                      + self.sim_data.grad_r_x_ij[:,:,1]*(self.sim_data.H_r_x_ij[:,:,i,1]-self.sim_data.grad_r_x_ij[:,:,1]*self.sim_data.grad_r_x_ij[:,:,i]/self.sim_data.r_x_ij)\
                            ) for i in range(2)] ).transpose((1,2,0))
        #part of  second term third part of product rule of Eq. (5) in version 08.10.2021] note ||**2 is multiply out
        # grad(grad c / c)
        self.eq5_term1_2_0 = self.sim_data.H_r_x_ij/self.sim_data.r_x_ij[:,:,np.newaxis,np.newaxis]
        self.eq5_term1_2_0[:,:,0,0]  -=   self.sim_data.grad_r_x_ij[:,:,0]*self.sim_data.grad_r_x_ij[:,:,0]/(self.sim_data.r_x_ij*self.sim_data.r_x_ij)
        self.eq5_term1_2_0[:,:,1,1]  -=   self.sim_data.grad_r_x_ij[:,:,1]*self.sim_data.grad_r_x_ij[:,:,1]/(self.sim_data.r_x_ij*self.sim_data.r_x_ij)
        tmp =self.sim_data.grad_r_x_ij[:,:,0]*self.sim_data.grad_r_x_ij[:,:,1]/(self.sim_data.r_x_ij*self.sim_data.r_x_ij)
        self.eq5_term1_2_0[:,:,0,1]  -=  tmp
        self.eq5_term1_2_0[:,:,1,0]  -=  tmp

    def decision(self,L,*args):
        r_mean = np.sum(np.multiply(L,self.sim_data.r_x_ij)) # mean rate 'r0' (at center of agent) [CAUTION: not multiplied by bin area because likelihood already represents probabilities per spatial bin]
        mean_grad_r_x = np.sum(np.multiply(L,self.sim_data.grad_r_x_ij[:,:,0]))  #mean grad r0:<grad r0> x component
        mean_grad_r_y = np.sum(np.multiply(L,self.sim_data.grad_r_x_ij[:,:,1]))  #mean grad r0 <grad r0> y component
        rfinite_mean = np.sum(np.multiply(L,self.sim_data.rfinite_x_ij))
       
        # first term of Eq. (5) in version 08.10.2021]
        TMP = self.sim_data.log_rfinite_x_ij - math.log(rfinite_mean)
        t0_x = np.sum(   L*(  TMP * self.sim_data.grad_rfinite_x_ij[:,:,0]) ,axis=(0,1))
        t0_y = np.sum(   L*(  TMP * self.sim_data.grad_rfinite_x_ij[:,:,1]) ,axis=(0,1))

        
        # ||**2 part of Eq. (5) in version 08.10.2021]
        TMPx = self.sim_data.grad_r_x_ij[:,:,0]/self.sim_data.r_x_ij - mean_grad_r_x/r_mean
        TMPy = self.sim_data.grad_r_x_ij[:,:,1]/self.sim_data.r_x_ij - mean_grad_r_y/r_mean
        t_norm = TMPx*TMPx+TMPy*TMPy

        # second term first part of product rule of Eq. (5) 'grad r * ||**2' in version 08.10.2021]
        t1_0_x = self.sim_data.grad_rfinite_x_ij[:,:,0]*t_norm
        t1_0_y = self.sim_data.grad_rfinite_x_ij[:,:,1]*t_norm
        #  eq5_term1_2_0 : grad(grad (c) / c) outer product
        #  eq5_term1_1   : 'grad (B**2): grad (nabla c/c)'
        # second third part of product rule of Eq. (5) 'grad (2A*B)' in version 08.10.2021] note ||**2 = A**2+B**2+2A*B
        #  grad(A**2 )=0
        t1_2_x = -2*self.eq5_term1_2_0[:,:,0,0]*mean_grad_r_x/r_mean + -2*self.eq5_term1_2_0[:,:,0,1]*mean_grad_r_y/r_mean
        t1_2_y = -2*self.eq5_term1_2_0[:,:,1,0]*mean_grad_r_x/r_mean + -2*self.eq5_term1_2_0[:,:,1,1]*mean_grad_r_y/r_mean
        # second term of Eq. (5) in version 08.10.2021] with correction to match Andreas note ||**2 = A**2+B**2+2A*B
        t1_x = self.sim_data.agent_size_2/4*(np.sum(L*\
            (t1_0_x +  self.sim_data.rfinite_x_ij*(self.eq5_term1_1[:,:,0]+t1_2_x) ) \
            ,axis=(0,1)) )
        t1_y = self.sim_data.agent_size_2/4*(np.sum(L*\
            (t1_0_y +  self.sim_data.rfinite_x_ij*(self.eq5_term1_1[:,:,1]+t1_2_y) ) \
            ,axis=(0,1)) )
        
        
        move_direction_x = - t0_x-t1_x
        move_direction_y = - t0_y-t1_y
        non_normalized = np.array([move_direction_x,move_direction_y])
        leng = math.sqrt(move_direction_x*move_direction_x + move_direction_y*move_direction_y)
        move_direction_x /= leng
        move_direction_y /= leng
        return np.array([move_direction_x,move_direction_y])


class Infotaxis_v(Infotaxis_ben):
    """
    Infotaxis to test the influence of an explotation term
    """
        
    def __init__(self,sim_data,lim_v0,lim_v1):
        super().__init__(sim_data)
        self.lim_v0 = lim_v0
        self.lim_v1 = lim_v1

    def decision(self,L,*args):
        r_mean = np.sum(np.multiply(L,self.sim_data.r_x_ij)) # mean rate 'r0' (at center of agent) [CAUTION: not multiplied by bin area because likelihood already represents probabilities per spatial bin]
        mean_grad_r_x = np.sum(np.multiply(L,self.sim_data.grad_r_x_ij[:,:,0]))  #mean grad r0:<grad r0> x component
        mean_grad_r_y = np.sum(np.multiply(L,self.sim_data.grad_r_x_ij[:,:,1]))  #mean grad r0 <grad r0> y component
        rfinite_mean = np.sum(np.multiply(L,self.sim_data.rfinite_x_ij))
       
        # first term of Eq. (5) in version 08.10.2021] without - sign
        TMP = self.sim_data.log_rfinite_x_ij - math.log(rfinite_mean)
        t0_x = np.sum(   L*(  TMP * self.sim_data.grad_rfinite_x_ij[:,:,0]) ,axis=(0,1))
        t0_y = np.sum(   L*(  TMP * self.sim_data.grad_rfinite_x_ij[:,:,1]) ,axis=(0,1))

        
        # ||**2 part of Eq. (5) in version 08.10.2021]
        TMPx = self.sim_data.grad_r_x_ij[:,:,0]/self.sim_data.r_x_ij - mean_grad_r_x/r_mean
        TMPy = self.sim_data.grad_r_x_ij[:,:,1]/self.sim_data.r_x_ij - mean_grad_r_y/r_mean
        t_norm = TMPx*TMPx+TMPy*TMPy

        # second term Eq. (5) first part of product rule of  'grad r * ||**2' in version 08.10.2021]
        t1_0_x = self.sim_data.grad_rfinite_x_ij[:,:,0]*t_norm
        t1_0_y = self.sim_data.grad_rfinite_x_ij[:,:,1]*t_norm
        # "eq5_term1_2_0" : 
        # second term Eq. (5) first and third part of product rule 'grad (2A*B)' in version 08.10.2021] note ||**2 = A**2+B**2+2A*B
        t1_2_x = -2*self.eq5_term1_2_0[:,:,0,0]*mean_grad_r_x/r_mean + -2*self.eq5_term1_2_0[:,:,0,1]*mean_grad_r_y/r_mean
        t1_2_y = -2*self.eq5_term1_2_0[:,:,1,0]*mean_grad_r_x/r_mean + -2*self.eq5_term1_2_0[:,:,1,1]*mean_grad_r_y/r_mean
        # second term of Eq. (5) in version 08.10.2021] eq5_term1_1 : 'grad (B**2)' with correction to match Andreas note ||**2 = A**2+B**2+2A*B
        t1_x = self.sim_data.agent_size_2/4*(np.sum(L*\
            (t1_0_x +  self.sim_data.rfinite_x_ij*(self.eq5_term1_1[:,:,0]+t1_2_x) ) \
            ,axis=(0,1)) )
        t1_y = self.sim_data.agent_size_2/4*(np.sum(L*\
            (t1_0_y +  self.sim_data.rfinite_x_ij*(self.eq5_term1_1[:,:,1]+t1_2_y) ) \
            ,axis=(0,1)) )
        
        
        move_direction_x = - t0_x-t1_x
        move_direction_y = - t0_y-t1_y
        #non_normalized = np.array([move_direction_x,move_direction_y])
        leng = math.sqrt(move_direction_x*move_direction_x + move_direction_y*move_direction_y)
        self.g = leng
        
        centeridx = self.sim_data.center / self.sim_data.space_discretization_step
        if centeridx%1 < 0.1 and centeridx%1 > 0.9:
            idx = int(np.round(centeridx))
            gradx = (L[idx+1,idx] - L[idx-1,idx] ) * self.sim_data.space_discretization_step
            grady = (L[idx,idx+1] - L[idx,idx-1] ) * self.sim_data.space_discretization_step
        else:
            idx = int(np.floor(centeridx))
            gradx = (L[idx+1,idx] - L[idx-1,idx] + L[idx+1,idx+1] - L[idx-1,idx+1] ) * 0.5 * self.sim_data.space_discretization_step
            grady = (L[idx,idx+1] - L[idx,idx-1] + L[idx+1,idx+1] - L[idx+1,idx-1] ) * 0.5 * self.sim_data.space_discretization_step
        self.h =  math.sqrt(gradx*gradx + grady*grady) * math.pi * self.sim_data.agent_size_2
        return np.array([move_direction_x/leng,move_direction_y/leng])

    def log(self):
        return self.g,self.h

class maxLog(decision_base):
    """
    maximum-likelihood method
    """
    def calc_d_matrices(self):
        self.x_ij,self.y_ij = xy_ij(self.sim_data.N,self.sim_data.space_discretization_step)

    def decision(self,L,*args):
        xidx,yidx = np.unravel_index(np.argmax(L),L.shape)
        move_direction_x = self.x_ij[xidx,yidx]
        move_direction_y = self.y_ij[xidx,yidx]
        non_normalized = np.array([move_direction_x,move_direction_y])
        leng = math.sqrt(move_direction_x*move_direction_x + move_direction_y*move_direction_y)
        move_direction_x /= leng
        move_direction_y /= leng
        return np.array([move_direction_x,move_direction_y])






