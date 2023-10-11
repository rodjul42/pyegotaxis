from random import seed
import sys

import numpy as np
import numpy.linalg as algebra
import math
import copy
# from scipy import stats
# import seaborn as sns    
import time

import logging
logger = logging.getLogger(__name__)


# import user libraries
from .functions import *
from .simdata import *
from . import functions3dDiff as functionsC
def draw_events(sim_data, dt, source):
    '''
    Draw random event for each  receptor
    sim_data    : class instance of  with all paramters and matrices
    dt          : time step
    source      : position of the source
    # loop over all directions of receptors on agent's circumference
    # vectorize loop
    '''
    this_rates = np.zeros(sim_data.directions.shape[0])
    functionsC.fieldv(sim_data.lambdaReal_, source - sim_data.agent_sizeReal*sim_data.directions,this_rates,sim_data.space_discretization_step) 
    this_rates  /= sim_data.N_angles # Note: normalizaton by 'N_angles', thus ensuring that the total binding rate is indepedendent of 'N_angle'
    dm_possion =  np.random.poisson(this_rates*dt) # Poisson distributed vectorial events
    return dm_possion.sum(), (sim_data.directions.T * dm_possion).sum(axis=1)
    
def no_events(sim_data, dt, source):
    return 0,np.array([0,0])



def run_sim(sim_data,decision,predict,update_step,get_events=draw_events,plot=False):
    """
    start the loop for the egotaxis
    sim_data    : class instance of sim_data with all paramters and matrices
    decision    : class instance with the decision step. Must have the function decision.decision(likelyhood)
    predict     : class instance for the prediction step. Must have predict.update_source(source,sim_data,dt)
                  and predict.update_likelyhood(L,dx,dt,sim_data)  
    update_step : update routine for the likelyhood update_step(norm_dm,dm,L_predict,sim_data)
                  norm_dm  : number of events
                  dm       : average event direction
                  L_predict: predicted likelyhood
                  sim_data : sim_data
    plot        : real time plotting with gr
    """
    np.random.seed(sim_data.seed) # initialize random number generator
    L = sim_data.Linit.copy()
    # initialize time series
    sampling = 0 # counter for saving results
    sampling2 = 0 # counter for saving results rarley
    
    S_t = []
    Sphi_t = []
    Sr_t = []
    
    p_lost_t = [] # [] lost probability
    dm_t = [] # list of events with direction
    norm_dm_t = [] # list of events
    dm_tmp = np.array([0.,0.]) 
    norm_dm_tmp=0
    source_t = [] # [m] target position
    list_t = [] # [s] list of times
    decisions_t = [] # # Data from decision step
    predict_t = []  # Data from predict step
    update_t = []  # Data from update step
    likelihood_history = []

    source = np.copy(sim_data.source)
    distance = math.sqrt(source[0]*source[0]+source[1]*source[1]) # [m] distance from target
    reached = False
    missed = False
    t_end = np.nan
    if sim_data.boundary == 'a':
        p_lost = np.sum(L[sim_data.nogo_area])
        L[sim_data.nogo_area] = 0
        #print('p_lost initial',p_lost)
    else:
        p_lost=0


    # Loop over time steps --------------------------------------------------------
    tic = time.time() # 26sec for N=200, T=1


    t=0
    dt = sim_data.dt
    minL = sim_data.minL*np.ones_like(L)
    while (t<sim_data.T):   
        
        # Decision step (based on current likelihood map 'L', chosen strategy, precomputed binding rate field and its derivaties)    
        move_direction  = decision.decision(L,source)
        # Move the target in the egocentric map of the agent (in opposite direction) and update the likelyhood accordingly
        dx = sim_data.speed * move_direction  # [m] displacement vector of agent
        #print(move_direction)

        # update the position of the source in egocentric map, may include diffusion
        source = predict.update_source(source,dx,dt)

  
        
        distance = math.sqrt(source[0]*source[0]+source[1]*source[1]) # [m] distance from target
        if distance < sim_data.target_size : # target reached?
            #a('source reached!')
            reached = True
            t_end = t
            break
        if distance > sim_data.max_dist: # agent left search domain?
            if sim_data.boundary=='a':
                #print('source missed!')
                missed = True
                t_end = t
                break
            elif sim_data.boundary=='g':
                #print('source missed!')
                missed = True
                t_end = t
                break
            elif sim_data.boundary=='r':
                #refelct the source at the boundary
                #as an prroximation the curvature of the bounsary is ingnored locally (the radius of the boundary is large compared dx)
                missed = True
                outsite = distance - sim_data.max_dist
                source_norm = source / distance
                source -= source_norm * 2*outsite
                #update dx also since it is needed for the prediction step
                dx += source_norm * 2*outsite    
            elif sim_data.boundary=='g': # run bc as reflecting but save as absorbing and also remove likelyood
                missed = True
                t_end = t
                #refelct the source at the boundary
                #as an prroximation the curvature of the bounsary is ingnored locally (the radius of the boundary is large compared dx)
                outsite = distance - sim_data.max_dist
                source_norm = source / distance
                source -= source_norm * 2*outsite
                #update dx also since it is needed for the prediction step
                dx += source_norm * 2*outsite 
            else:
                raise NotImplementedError
        # Prediction step, accounting for convection (due to active motion of the agent) and diffusion (due to motility noise of the agent)
        L_predict = predict.update_likelyhood(L,dx,dt)
    
        #get the new input from the receptors
        norm_dm,dm = get_events(sim_data, dt, source)       
        dm_tmp += dm # keep a record    
        norm_dm_tmp += norm_dm
        # update Likelihood bases on the events
        L = update_step.update(norm_dm,dm,L_predict,dt,sim_data)
        
        # Regularization for numerical stability: ensure that all likelihood values are larger than 'min_L'    
        L = np.maximum( L, minL )
        
        if sim_data.boundary == 'a':
            L /= np.sum(L) # normalize total likelihood to one
            tmp = np.sum(L[sim_data.nogo_area])  # calulate the lost probablity 
            p_lost += tmp*(1-p_lost)  # keep track of lost probability and account for the fact actually sum(L) + p_lost is one
            L[sim_data.nogo_area] = sim_data.minL  #delete lost probability
            L /= 1-tmp    #normalizes new L to one 
        elif sim_data.boundary == 'b':  #old version with non normalized L
            L *= (1-p_lost)/np.sum(L) 
            tmp = np.sum(L[sim_data.nogo_area])  # calulate the lost probablity 
            p_lost += tmp  # keep track of lost probability and account for the fact actually sum(L) + p_lost is one
            L[sim_data.nogo_area] = sim_data.minL
        elif  sim_data.boundary == 'r':
            L[sim_data.nogo_area] = sim_data.minL  #delete lost probability
            L /= np.sum(L) # normalize total likelihood to one
        else:
            raise NotImplementedError

        # keep a record (every 'sampling_interval' time steps)        
        if sampling % sim_data.sampling_interval == 0:
            #print(f'progress: {100*t/self.T:.2f}%, distance: {distance:0.3f}')
            #print(algebra.linalg.norm(move_direction-move_directionn),move_direction-move_directionn)
            S_t.append( entropy( L ) )
            Sphi_t.append( entropy_phi(L,sim_data) )
            Sr_t.append( entropy_r(L,sim_data) )

            p_lost_t.append( p_lost )
            # keep a record        
            list_t.append(t)
            source_t.append(np.copy(source))
            dm_t.append(np.copy(dm_tmp))
            dm_tmp *= 0
            norm_dm_t.append(norm_dm_tmp)
            norm_dm_tmp = 0
            decisions_t.append(decision.log())
            if update_step.__class__.__name__=='ben_INFO_delta' and predict.__class__.__name__=='predict_adv_diffrot_kernel_INFO_delta':
                if sampling==0:
                    predict.Es[:] = predict.Es[0]/sim_data.sampling_interval
                    update_step.Es[:] = update_step.Es[0]/sim_data.sampling_interval
                    predict.Entropy_last_step = predict.Es[0,:3]
                    tmp = copy.deepcopy(predict.Es)
                    predict_t.append(predict.log(predict.Es))
                    predict.Entropy_last_step = update_step.Es[-1,:3]*sim_data.sampling_interval
                    update_t.append(update_step.log(tmp))
                    
                else:
                    tmp = copy.deepcopy(predict.Es)
                    predict_t.append(predict.log(update_step.Es))
                    update_t.append(update_step.log(tmp))
            else:
                predict_t.append(predict.log(None))
                update_t.append(update_step.log(None))
            #self.likelihood_history.append(L)
            if sampling2 % sim_data.sampling2_interval == 0:
                likelihood_history.append(L.copy())
            sampling2 += 1 # step counter for saving results

        sampling += 1 # step counter for saving results

        t += dt



    time_elapsed = time.time() - tic
    #print('time elapsed:', time_elapsed, 's')    
    if decisions_t[0] is None:
        decisions_t = None
    else:
        decisions_t = np.array(decisions_t)

    predict_t = predict.process_log(predict_t)    
    update_t = update_step.process_log(update_t)    
    

    log = SIMLOG(
        time = np.array(list_t),
        entropy = np.array(S_t),
        entropy_phi = np.array(Sphi_t),
        entropy_r = np.array(Sr_t),
        
        p_lost = np.array(p_lost_t),
        events = np.array(dm_t),
        Nevents = np.array(norm_dm_t),
        source = np.array(source_t),
        decisions = decisions_t,
        predict = predict_t,
        update = update_t,
        likelihood = likelihood_history,
        lastL = L,
        initL = sim_data.Linit,
        stats = {'reached':reached,'missed':missed,'p_lost':p_lost,'time':t_end,
            'decision':decision.__class__.__name__,
            'update_step':update_step.__class__.__name__,
            'predict':predict.__class__.__name__},
        parameter = sim_data._asdict()
       )
    return log



