from cmath import nan
import os
import re
import numpy as np
from scipy.signal import find_peaks
from collections import namedtuple
import  scipy.stats
from scipy import optimize


import powersmooth as psmooth
import pandas as pd

from . import utils

import logging
logger = logging.getLogger(__name__)

UpperQuantile = 0.84
LowerQuantile = 0.16
maxentropy_phi = 6.64379

SIMLOG_tcsc = namedtuple('simlog_tcsc',['time','entropy','entropy_phi','entropy_r','entropytc','entropytc_phi','entropytc_r','entropysc','entropysc_phi','entropysc_r',
                                    'p_lost','events','Nevents','source','decisions','predict','update','likelihood','initL','lastL','stats','parameter'] )
SIMLOG = namedtuple('simlog_v4',['time','entropy','entropy_phi','entropy_r','p_lost','events','Nevents','source','decisions','predict','update','likelihood','initL','lastL','stats','parameter'] )
SIMLOG_v3 = namedtuple('simlog_v3',['time','entropy','entropy_phi','entropy_r','p_lost','events','Nevents','source','decisions','likelihood','initL','lastL','stats','parameter'] )
SIMLOG_v2 = namedtuple('simlog_v2',['time','entropy','entropy_phi','entropy_r','events','Nevents','source','decisions','likelihood','initL','lastL','stats','parameter'] )
SIMLOG_old2 = namedtuple('simlog',['time','entropy','events','source','distance','move_direction','initL','lastL','stats','parameter'] )
SIMLOG_old = namedtuple('simlog', ['time','entropy','events','source','distance','move_direction','lastL','reached','missed','p_lost'] )


def dict_to_SIMLOG(sim_dict):
    """
    convert dict to SIMLOG
    adds NANS to missing values.
    """
    keys = list(sim_dict.keys())
    if 'initL' in keys and 'entropy_r' not in keys:  #convert old2 data to new
        NANS = np.ones_like(sim_dict['entropy'])*np.nan
        sim_dict['entropy_phi'] = NANS
        sim_dict['entropy_r'] = NANS
        sim_dict['Nevents'] = NANS
        sim_dict['decisions'] = None
        sim_dict['likelihood'] = []
        del sim_dict['distance']
        del sim_dict['move_direction']
        return SIMLOG(**sim_dict)
    try:
        sim_dict['Nevents']
    except KeyError:
        NANS = np.ones_like(sim_dict['entropy'])*np.nan
        sim_dict['Nevents'] = NANS
    try:
        sim_dict['p_lost']
    except KeyError:
        NANS = np.ones_like(sim_dict['entropy'])*np.nan
        sim_dict['p_lost'] = NANS
    try:
        sim_dict['parameter']['sampling2_interval']
    except KeyError:
        sim_dict['parameter']['sampling2_interval'] = 100

    try:
        sim_dict['predict']
    except KeyError:
        sim_dict['predict'] = None

    try:
        sim_dict['update']
    except KeyError:
        sim_dict['update'] = None

    try:
        sim_dict['stats']['duration']
    except KeyError:
        sim_dict['stats']['duration'] = np.nan
    return SIMLOG(**sim_dict)


def load_data(path, name = None, parameter = None,skipTest=False,roundp=10,phi0=False):
    """
    skipTest : skip test of paramters
    parameter : check if parmeter matches to the one loaded
    roundp : round to this decimal number for check 
    phi0 : ignore this angle
    load data and convert to SIMDATA
    """
    if name is None:
        name = 'arr_0'
    with np.load(path,allow_pickle=True) as f:
        res = f[name]
    ret = []
    for i_n,i in enumerate(res):
        sim_data = dict_to_SIMLOG(i)
        if phi0 and sim_data.parameter['phi0'] == phi0:
            continue
        if parameter is not None and skipTest is False:
            for name,value in parameter.items():
                if name == "lam":
                    name = "lambda_"
                if np.round(sim_data.parameter[name],roundp) !=  value:
                    raise ValueError(f'The parameter {name} from the sim ({sim_data.parameter[name]}) does not mathc the one ({value}) from the file name {path}')
        sim_data.parameter['filename'] = f'{path}#{i_n}'
        ret.append(sim_data)
    return ret


def load_single(file,name = 'arr_0', parameter = None,skipTest=False,roundp=10,phi0=False,rmL=True):
    """
    skipTest : skip test of paramters
    parameter : check if parmeter matches to the one loaded
    roundp : round to this decimal number for check 
    phi0 : ignore this angle
    rmL : remove likelihood
    load data and convert to SIMDATA
    """
    with np.load(file,allow_pickle=True) as f:
        res = f[name]
    data = res.item()
    if rmL:
        data['likelihood'] = []
    sim_data = dict_to_SIMLOG(data)
    if phi0 and sim_data.parameter['phi0'] == phi0:
        raise ValueError('phi0 does not match')
    if parameter is not None and skipTest is False:
        for pname,value in parameter.items():
            if pname == "lam":
                pname = "lambda_"
            if pname == "gamma":
                pname = "gammaReal"
            if np.round(sim_data.parameter[pname],roundp) !=  value:
                raise ValueError(f'The parameter {pname} from the sim ({sim_data.parameter[pname]}) does not mathc the one ({value}) from the file name {file}')
    sim_data.parameter['filename'] = file
    return sim_data

def load_data_folder(path, name = None, parameter = None,skipTest=False,roundp=10,phi0=False,rmL=True):
    """
    skipTest : skip test of paramters
    parameter : check if parmeter matches to the one loaded
    roundp : round to this decimal number for check 
    phi0 : ignore this angle
    rmL : remove likelihood
    load from folder and convert to SIMDATA
    """
    if name is None:
        name = 'arr_0'

    ret = []
    files = os.listdir(path)
    for file in files:
        try:
            ret.append(load_single(os.path.join(path,file),name = name, parameter = parameter,skipTest=skipTest,roundp=roundp,phi0=phi0,rmL=rmL))
        except Exception as inst:
            print(file)
            print(inst)
            continue
        
    return ret


def read_filename(file,typs):
    '''
    typs: paramters names to get from file name
    get paramters from filename 
    '''
    return re.sub("_".join([typ+r"([0-9.+-e]*|inf)" for typ in typs])+"_([0-9]*).npz","#,#".join(["\\"+str(i+1) for i in range(len(typs)+1)]), file).split('#,#')


def read_foldername(file,typs):
    '''
    typs: paramters names to get from folder name
    get paramters from folder name 
    '''
    return re.sub("_".join([typ+r"([0-9.+-e]*|inf)" for typ in typs]),"#,#".join(["\\"+str(i+1) for i in range(len(typs))]), file).split('#,#')


def get_idxs(path_folder,typs,roundp=10):
    """
    read all files from folder and extracat parameter from file names
    typs: paramters names to get from folder name
    """
    files = os.listdir(path_folder)
    parameter = pd.DataFrame()
    # { name:val for name,val in zip(typs+["num"],values) }
    for file in files:
        
        if os.path.isfile(os.path.join(path_folder,file)):
            values = read_filename(file,typs)
            idx = pd.MultiIndex(levels=[[np.round(float(i),roundp)] if i!="inf" else np.inf for i in values[:-1]],
                                codes=[[0] for i in range(len(typs))],
                                name= [i   for i in typs])
            try:
                parameter.loc[idx[0],"file"] += [file]
                parameter.loc[idx[0],"folder"] += []
            except KeyError:
                tmp = pd.DataFrame([[[file],[]]],index=idx,columns=["file","folder"])
                parameter = pd.concat([parameter,tmp],verify_integrity=True,sort=True).sort_index()
        else:
            values = read_foldername(file,typs)
            idx = pd.MultiIndex(levels=[[np.round(float(i),roundp)] if i!="inf" else np.inf for i in values],
                                codes=[[0] for i in range(len(typs))],
                                name= [i   for i in typs])
            try:
                parameter.loc[idx[0],"folder"] += [file]
                parameter.loc[idx[0],"file"] += []
            except KeyError:
                tmp = pd.DataFrame([[[file],[]]],index=idx,columns=["folder","file"])
                parameter = pd.concat([parameter,tmp],verify_integrity=True,sort=True).sort_index()
            
    return parameter





def get_all_stats(path_folder,all_flies,skipTest=False,phi0=False,**args):
    """
    calulate stats for all trajectories and put all in one big dataframe
    path_folder : folder of files
    all_flies   : list of files see get_idxs
    skipTest    : skip test if file names match parameters
    phi0        : ingore this angle
    args        : argd for  function
    """
    names = all_flies.index.names
    all_stats = []
    index = []
    for idx,row in all_flies.iterrows():
        names = all_flies.index.names
        idx_dict = {n:i for n,i in zip(names,idx)}
        data = []       
        file_names = [] 
        for file in row["file"]:
            try:
                data += load_data(os.path.join(path_folder,file),parameter=idx_dict,skipTest=skipTest,phi0=phi0)
                file_names.append( file )
            except Exception as inst:
                print(inst,file) 
                raise inst
        for file in row["folder"]:
            try:
                data += load_data_folder(os.path.join(path_folder,file),parameter=idx_dict,skipTest=skipTest,phi0=phi0)
                file_names.append( file )
            except Exception as inst:
                print(inst,file) 
                raise inst
        if len(data) == 0:
            logger.warning(f"Sim empty {os.path.join(path_folder,file)}")
            continue 
        
        sl = []
        for i in data:
            x,y = utils.calc_lab_Drot(i.source, i.predict).T
            sl.append([x,y])
        tmp = get_stats(data,**args)
        tmp["sourcelab"] = sl
        tmp["time"] = data[np.argmax([len(i.time) for i in data])].time
        tmp["entropy"] = [i.entropy  for i in data]
        tmp["stats"] = [i.stats for i in data]
        tmp["entropy_phi"] = [i.entropy_phi for i in data]
        tmp["parameter"] = [i.parameter for i in data]
        tmp["Nevents"] = [i.Nevents for i in data]
        tmp['file'] = file_names
        all_stats.append(tmp)
        index.append(list(idx))


    stats = pd.DataFrame(all_stats,index=pd.MultiIndex.from_tuples(index,names=names))
    return stats



def get_stats(sims,FirstTurn=False,frame='lab',w1_smooth=1000,height=15,QUANTILS = [0.25,0.5,0.75],
                Nbootstrap=1000,
                min_N_traj=101,
                **args):
    '''
    calc stats for sim
    FirstTurn : do the first turn alaysis, only work if Drot=0
        frame : lab or egocentric reference frame
        w1_smooth : wight for powersmooth 1 order
        height : Height for first peak detection in absolute curveture

    QUANTILS : quantils to calulate
    Nbootstrap : number of bootstrap for errors
    min_N_traj: minimal number of trajetories for Iphi_average
    '''

    def fit_f(p,x):
        return np.where(x<p[0],p[1]*x,p[1]*p[0])

    def fit_m(p,x,d):
        return np.sqrt(np.sum((fit_f(p,x) - d)**2))


    statistics = {}
    if FirstTurn:
        idxswitch_c = []

        for i_n,sim in enumerate(sims):
            if not sim.stats['reached']:
                pass        

            if frame=='lab':
                x,y = -sim.source.T
                x0 = -sim.parameter['source'][0]
                y0 = -sim.parameter['source'][1]
            elif frame=='ego':
                x,y = sim.source.T
                x0 = sim.parameter['source'][0]
                y0 = sim.parameter['source'][1]
            
            ps = psmooth.multi_smooth(len(x),[1],[w1_smooth])
            xs = ps(x)
            ys = ps(y)
            
            curv = utils.calc_curv(xs,ys,sim.time)
            acurv = np.abs(curv)
            
            peaks2,_ = find_peaks(acurv,height =height)
            if len(peaks2)>0:
                for pi in peaks2:
                    p = pi + 1
                    if (maxentropy_phi - sim.entropy_phi[p])>0.1:
                        break
                idxswitch_c.append( p )
            else:
                idxswitch_c.append(np.nan)
        
        
        statistics['idxswitch_c'] = idxswitch_c


    ########## calc average Iphi and Iphi star
    avg_Iphi = []
    for i_n,sim in enumerate(sims):
        avg_Iphi.append(pd.Series(maxentropy_phi - sim.entropy_phi,index=sim.time))
    avg_Iphi = pd.concat(avg_Iphi,axis=1)
    idx_avg_Iphi_maxT = np.where(avg_Iphi.notna().sum(axis=1) < min_N_traj )[0][0]
    avg_Iphi = avg_Iphi.mean(axis=1)
    res = scipy.optimize.minimize(fit_m,[1,1],args=(avg_Iphi.index[:idx_avg_Iphi_maxT],
                                                    avg_Iphi.values[:idx_avg_Iphi_maxT]))
    Iphi_inf = res['x'][0]*res['x'][1]

    #####
    ## Calc missed and reached
    #####
    N_found_target = np.sum([sim.stats['reached'] for sim in sims])
    N_missed_target = np.sum([sim.stats['missed'] for sim in sims])

    time_to_miss_target = [sim.stats['time'] if sim.stats['missed'] else np.nan for sim in sims]
    time_to_find_target = [sim.stats['time'] if sim.stats['reached'] else np.nan for sim in sims]
    average_events_to_find = np.mean([sim.Nevents.sum() for sim in sims if sim.stats['reached']] )
    p_lost  =  np.sum([sim.stats['p_lost'] for sim in sims])
    N_sim = len(sims)

    #####
    ## Calc moments and errors
    #####
    if N_found_target>0:
        time_to_find_target_noNaN = [i for i in  time_to_find_target if np.isfinite(i)]
        desc = scipy.stats.describe(time_to_find_target_noNaN)
        fpt_moments = np.zeros((5,3))
        fpt_moments[1:5,0] = [desc.mean,np.sqrt(desc.variance),desc.skewness,desc.kurtosis]
        
        
        moments = np.zeros((Nbootstrap,4))
        for bi in range(Nbootstrap):
            bdata = np.random.choice(time_to_find_target_noNaN, size=len(time_to_find_target_noNaN), replace=True)
            desc = scipy.stats.describe(bdata)
            moments[bi] = [desc.mean,np.sqrt(desc.variance),desc.skewness,desc.kurtosis]
        fpt_moments[1:,1] = np.quantile(moments,UpperQuantile,axis=0) 
        fpt_moments[1:,2] = np.quantile(moments,LowerQuantile,axis=0) 
        
        fpt_moments[0,0] = np.mean(time_to_find_target_noNaN)
        fpt_moments[0,1] = np.quantile(time_to_find_target_noNaN,UpperQuantile,axis=0) 
        fpt_moments[0,2] = np.quantile(time_to_find_target_noNaN,LowerQuantile,axis=0) 

        fpt_quantils = np.zeros((len(QUANTILS),3))
        for i_n,i in enumerate(QUANTILS):
            fpt_quantils[i_n,0] = utils.nanquantil(time_to_find_target,i)
        quantils = np.zeros((Nbootstrap,len(QUANTILS)))
        for bi in range(Nbootstrap):
            bdata = np.random.choice(time_to_find_target, size=len(time_to_find_target), replace=True)
            for i_n,i in enumerate(QUANTILS):
                quantils[bi,i_n] = utils.nanquantil(bdata,i)
        fpt_quantils[:,1] = np.quantile(quantils,UpperQuantile,axis=0) 
        fpt_quantils[:,2] = np.quantile(quantils,LowerQuantile,axis=0) 

    else:
        fpt_moments = np.zeros((4,3))*np.nan



    statistics.update( {'P_reached':N_found_target*1./N_sim,'N_reached':N_found_target,'time_to_find_target':time_to_find_target,
            'P_missed':N_missed_target*1./N_sim,'N_missed':N_missed_target,
            'time_to_miss_target':time_to_miss_target,'average_events_to_find':average_events_to_find,
            'fpt_mean':fpt_moments[0,0],'fpt_mean_u':fpt_moments[0,1],'fpt_mean_l':fpt_moments[0,2],
            'fpt_std':fpt_moments[1,0],'fpt_std_u':fpt_moments[1,1],'fpt_std_l':fpt_moments[1,2],
            'fpt_skew':fpt_moments[2,0],'fpt_skew_u':fpt_moments[2,1],'fpt_skew_l':fpt_moments[2,2],
            'fpt_kurosis':fpt_moments[3,0],'fpt_kurosis_u':fpt_moments[3,1],'fpt_kurtosis_l':fpt_moments[3,2],
            'avg_Iphi':avg_Iphi.values,'idx_avg_Iphi_maxT':idx_avg_Iphi_maxT,'Iphi_inf':Iphi_inf,
            'P_lost':p_lost*1./N_sim}
    )
    for i_n,i in enumerate(QUANTILS):
        statistics.update({f'fpt_{i}':fpt_quantils[i_n,0],f'fpt_{i}_u':fpt_quantils[i_n,1],f'fpt_{i}_l':fpt_quantils[i_n,2]})
    return statistics


def get_info_cumsum(data):   
    """
    Calulate the cummulative sum of the change of entropy from different sources
    Only work for data with the log of information from different sources 
    """
    tmp = []
    for i,sim in enumerate(data):
        Sall = -pd.concat([sim.predict[1],sim.update],axis=1)
        #Sall -= Sall.iloc[0]

        lastcum  =  Sall[['Sphi_rot','Sphi_adv','Sphi_tc','Sphi_sc','Sphi_t']].cumsum().iloc[-1]
        #lastcum.index = [i+"_csum" for i in lastcum.index]

        stats = pd.Series([1 if sim.stats['reached'] else 0],index=['reached'])
        last = pd.concat([lastcum,stats])
        tmp.append(last)
    
    tmp = pd.concat(tmp,keys=np.arange(len(data)),names=['sim'],axis=1).T
    return tmp.rename(columns={'Sphi_rot':'rot_csum','Sphi_adv':'adv_csum',
                      'Sphi_tc':'tc_csum','Sphi_sc':'sc_csum','Sphi_t':'full_csum'})


def add_info_stats(path_folder,all_flies,all_stats,saveDetails=None):
    '''
    inplace add information decomposition stats to all_stats
    Only work for data with the log of information from different sources 
    tc_csum : temporal comparrsion
    sc_csum : spatio comparrsion
    INFO_CUM: tc_csum / ( tc_csum + sc_csum )
    Added f at the end -> only use runs if the target is found
    INFO_CUMf_sc: sc_csumf / ( tc_csumf + sc_csumf )
    '''
    index = []
    i=0
    allstats = []
    index = []
    for idx,row in all_flies.iterrows():
        sims = load_data_folder(os.path.join(path_folder,row['folder'][0]))
        if len(sims) == 0:
            logger.warning(f"Sim empty {os.path.join(path_folder,row['folder'][0])}")
            continue
        stats =  get_info_cumsum(sims)
        
        index.append(idx)
        allstats.append(stats)
    mindex = pd.MultiIndex.from_tuples(index,names=all_flies.index.names)
    allstats = pd.concat(allstats,axis=0,keys=mindex)
    if saveDetails is not None:
        allstats.reset_index().to_feather(saveDetails)

    idx_names = all_flies.index.names
    allstats_grouped = allstats.groupby(idx_names).describe()
    allstats_groupedfound = (allstats[allstats['reached'] == 1]).groupby(idx_names).describe()
    summ = (allstats_grouped[('tc_csum','mean')] + allstats_grouped[('sc_csum','mean')])
    
    for idx,row in all_stats.iterrows():
        all_stats.loc[idx,'tc_csum'] = allstats_grouped.loc[idx,('tc_csum','mean')]
        all_stats.loc[idx,'sc_csum'] = allstats_grouped.loc[idx,('sc_csum','mean')]
        all_stats.loc[idx,'INFO_CUM']  = allstats_grouped.loc[idx,('tc_csum','mean')]/ \
                                        (allstats_grouped.loc[idx,('tc_csum','mean')]+allstats_grouped.loc[idx,('sc_csum','mean')])
        
        all_stats.loc[idx,'tc_csumf'] = allstats_groupedfound.loc[idx,('tc_csum','mean')]
        all_stats.loc[idx,'sc_csumf'] = allstats_groupedfound.loc[idx,('sc_csum','mean')]
        all_stats.loc[idx,'INFO_CUMf']  = allstats_groupedfound.loc[idx,('tc_csum','mean')]/ \
                                        (allstats_groupedfound.loc[idx,('tc_csum','mean')]+allstats_groupedfound.loc[idx,('sc_csum','mean')])
        
        all_stats.loc[idx,'INFO_CUMf_sc']  = allstats_groupedfound.loc[idx,('sc_csum','mean')]/ \
                                        (allstats_groupedfound.loc[idx,('tc_csum','mean')]+allstats_groupedfound.loc[idx,('sc_csum','mean')])
    return