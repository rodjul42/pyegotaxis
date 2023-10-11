from dataclasses import is_dataclass
import numpy as np
from scipy.ndimage import rotate
import pandas as pd
import matplotlib.pyplot as plt
colors =["#3f90da","#bd1f01", "#94a4a2", "#ffa90e",  "#832db6", "#a96b59", "#e76300", "#b9ac70", "#717581", "#92dadd"]
len_colors = len(colors)
from .tools import *



# FROM SEABORN
def _freedman_diaconis_bins(a):
    """Calculate number of hist bins using Freedman-Diaconis rule."""
    # From https://stats.stackexchange.com/questions/798/
    a = np.asarray(a)
    if len(a) < 2:
        return 1
    iqr = np.subtract.reduce(np.nanpercentile(a, [75, 25]))
    h = 2 * iqr / (len(a) ** (1 / 3))
    # fall back to sqrt(a) bins if iqr is 0
    if h == 0:
        return int(np.sqrt(a.size))
    else:
        return int(np.ceil((a.max() - a.min()) / h))



import collections
           
import statsmodels.api as sm
   

import bokeh
from bokeh.models import LinearColorMapper,  LogColorMapper,ColumnDataSource,ColorBar,HoverTool,CustomJS,TapTool,Line,Segment,Span,Range1d,Band,CustomJS, BasicTicker, ColorBar, Legend, LegendItem
from bokeh.models import BoxZoomTool,Title

from bokeh.layouts import column,row,gridplot
from bokeh.plotting import figure

def make_CDS(sim,Rframe,**args):
    data = {}
    params = getattr(sim,'parameter')
    if Rframe=='ego':
        x,y= getattr(sim,'source').T
    elif Rframe=='lab':
        if params['Drot'] == 0:
            x,y= getattr(sim,'source').T
            x = x*-1
            y = y*-1
        else:
            x,y = utils.calc_lab_Drot(sim.source,sim.predict).T 
    else:
        raise NotImplementedError
    d = np.sqrt(x*x+y*y)
    n = 'p'
    data.update({f'{n}x':x,f'{n}y':y,
                f'{n}x1':np.append(x[1:],x[-1]),f'{n}y1':np.append(y[1:],y[-1]),
                f'{n}n':d})    
    for n in ['time','Nevents']:
        data.update({n:getattr(sim,n)})
    for n in ['entropy','entropy_phi','entropy_r']:
        data.update({n:getattr(sim,n)})
    return ColumnDataSource(data,**args)



codet = """
    //console.log(s_likelihood[cb_data.source.name])
    //console.log(s_likelihood[cb_data.source.name].data.likelihood[0])
    let i = parseInt(cb_data.source.name)
    let j = GLOBAL['last_idx']
    
    if (i==j && !GLOBAL['reset']){
        GLOBAL['reset']=0
    }else{
        GLOBAL['reset']=0
    }
    let renderers = Ptrajectory.renderers
    /*
    for (let r=0;r<renderers.length;r++){
        if (r==i || GLOBAL['reset']){
            renderers[r].glyph.line_alpha = 1;
        }else{
            renderers[r].glyph.line_alpha = 0.001;
        }
    }
    */

    let s_idx = cb_data.source.selected.indices;
    if (s_idx.length==0){
        s_idx = cb_data.source.selected.line_indices;
    }
    
    // First update image to add the point to the traces
    renderers = Plikelihood.renderers
    let sourcesI = renderers[0].data_source;
    let sourceIp = renderers[1].data_source;
    let d2 = sourceIp.data

    let image = sourcesI.data.value[0]
    let I_idx = 0
    if (s_idx.length>0){

        I_idx = Math.round(s_idx[0]/###1###)
        
        if (I_idx>s_likelihood[cb_data.source.name].data.l.length-1){
            I_idx=s_likelihood[cb_data.source.name].data.l.length-1
        }
        if (Rframe=='lab'){
        let x0 = cb_data.source.data['px'][(I_idx)*###1###]
        let y0 = cb_data.source.data['py'][(I_idx)*###1###]
        sourcesI.data.x0[0] = x0 + ###x0###
        sourcesI.data.y0[0] = y0 + ###y0###
            d2['x'] = [x0]
            d2['y'] = [y0]
        }else{
            d2['x'] = [0.0]
            d2['y'] = [0.0]
        }
        for (let s=0;s<s_likelihood[cb_data.source.name].data.l[0].length;s++){
            image[s] = s_likelihood[cb_data.source.name].data.l[I_idx][s]
        }
    }
    sourcesI.change.emit()
    sourceIp.change.emit()

    renderers = Ptrajectory.renderers
    let sourcep = renderers[renderers.length-1].data_source;
    d2 = sourcep.data
    d2['x'] = []
    d2['y'] = []
    if (s_idx.length>0){
        //now add points and render
        d2['x'].push(cb_data.source.data['px'][(I_idx)*###1###])
        d2['y'].push(cb_data.source.data['py'][(I_idx)*###1###])
    }
    sourcep.change.emit();

    for (let subplot =0;subplot<subplots.length;subplot++){
        renderers = subplots[subplot].renderers
        if (parseInt(renderers[i].name) != i){
                return
            }
        if (j>-1){
            if (parseInt(renderers[j].name) != j){
                return
            }
            renderers[j].glyph.line_width = 1;
            renderers[j].glyph.line_alpha = 0.1;
        }

        if (GLOBAL['reset']==1){
            s_idx = []
        }else{
            renderers[i].glyph.line_width = 2;
            renderers[i].glyph.line_alpha = 1;
        }



        let sourcep = renderers[renderers.length-1].data_source;
        const d2 = sourcep.data
        d2['x'] = []
        d2['y'] = []
        if (s_idx.length>0){
            for (let s=0;s<s_idx.length;s++){
                d2['x'].push(cb_data.source.data.time[s_idx[s]])
                d2['y'].push(cb_data.source.data[subplots[subplot].name][s_idx[s]])
            } 
            //now add points and render
            d2['x'].push(cb_data.source.data.time[(I_idx)*###1###])
            d2['y'].push(cb_data.source.data[subplots[subplot].name][(I_idx)*###1###])
        }
        sourcep.change.emit();
    }
    renderers = Pspecial.renderers
    let sourcea = renderers[0].data_source;
    let sourceb = renderers[0].data_source;
    let sourcec = renderers[0].data_source;
    const a2 = sourcea.data
    a2['x'] = []
    a2['a'] = []
    a2['b'] = []
    a2['c'] = []
    for (let s=0;s<cb_data.source.data.time.length;s++){
        a2['x'].push(cb_data.source.data.time[s])
        a2['a'].push(cb_data.source.data['entropy'][s])
        a2['b'].push(cb_data.source.data['entropy_r'][s])
        a2['c'].push(cb_data.source.data['entropy_phi'][s])
    } 
    sourcea.change.emit()
    sourceb.change.emit()
    sourcec.change.emit()
    
    

    GLOBAL['last_idx'] = i
    """


codeh = """
    const indices = cb_data.index.line_indices
    if (indices.length>0){
    let i = parseInt(cb_data.renderer.data_source.name)
    let renderers = Pspecial.renderers
    let sourcea = renderers[0].data_source;
    let sourceb = renderers[0].data_source;
    let sourcec = renderers[0].data_source;
    const a2 = sourcea.data
    a2['x'] = []
    a2['a'] = []
    a2['b'] = []
    a2['c'] = []
    for (let s=0;s<cb_data.renderer.data_source.data.time.length;s++){
        a2['x'].push(cb_data.renderer.data_source.data.time[s])
        a2['a'].push(cb_data.renderer.data_source.data['entropy'][s])
        a2['b'].push(cb_data.renderer.data_source.data['entropy_r'][s])
        a2['c'].push(cb_data.renderer.data_source.data['entropy_phi'][s])
    } 
    sourcea.change.emit()
    sourceb.change.emit()
    sourcec.change.emit() 
    }
   

    """


def plot_bokeh(simulations,color_field='entropy_phi',Rframe='ego',typ=None):
    TOOLTIPS = [
            ("Time", "@time{0.}"),
            ("Events", "@Nevents{0}"),
            (color_field, "@"+color_field+"{0.0}"),
            ("Total Time", "$name")
        ]
    sim = simulations[0]  
    sp = 2*sim.parameter['l']/sim.parameter['N']
    c = (sim.parameter['N']-1)/2

    color_mapper = LinearColorMapper(palette='Turbo256', 
                    low =np.min([np.min(getattr(i,color_field)) for i in simulations]),
                    high=np.max([np.max(getattr(i,color_field)) for i in simulations]) )

    if typ is not None:
        simulations = [sim for sim in simulations if sim.stats[typ]]
    s_data = {idx:make_CDS(sim,Rframe=Rframe,name=str(idx)) for idx,sim in enumerate(simulations)}
    if sim.parameter['Drot'] == 0:
        s_likelihood = {idx:ColumnDataSource({'l':[l.T.ravel() for i_n,l in enumerate(sim.likelihood)]},name=str(idx)) for idx,sim in enumerate(simulations)}
    else:
        s_likelihood = {}
        for idx,sim in enumerate(simulations):
            tmp = []
            for i_n,l in enumerate(sim.likelihood):
                tmpd = l.copy()
                tmpd[90:110,0:20] = np.max(l)
                tmp.append(rotate(tmpd,sim.predict[i_n*sim.parameter['sampling2_interval']]/np.pi*180,reshape=False).T.ravel() )

            s_likelihood[idx] = ColumnDataSource(   {'l': tmp } ,name=str(idx)) 
         
    
    code = codet.replace('###1###',str(sim.parameter['sampling2_interval']))\
                .replace('###x0###',str(-c*sp-sp/2))\
                .replace('###y0###',str(-c*sp-sp/2))


    Ptrajectory = figure(width=600, height=500, title='Trajectory',tooltips=TOOLTIPS,match_aspect=True)
    Pentropy = figure(width=500, height=250, title='Entropy',name="entropy")
    PentropyR = figure(width=500, height=250, title='Entropy R',name="entropy_r")
    PentropyPhi = figure(width=500, height=250, title='Entropy Phi',name="entropy_phi")

    Pspecial = figure(width=600, height=250, title='Entropy combined',name="entropy2")


    subplots = [Pentropy,PentropyR,PentropyPhi]

    Plikelihood = figure(width=600, height=600, title='Likelihood',match_aspect=True)

    sourcesI = ColumnDataSource({'value': [sim.initL.T],'x0':[-c*sp-sp/2],'y0':[-c*sp-sp/2]})
    image = Plikelihood.image('value', source=sourcesI, x='x0', y='y0', dw=2*sim.parameter['l'], dh=2*sim.parameter['l'], palette="Spectral11", level="image")
    color_mapperL = LinearColorMapper(palette='Spectral11',  low =0,  high=1 )
    color_bar = ColorBar(color_mapper=color_mapperL,    location=(0,1))
    Plikelihood.add_layout(color_bar, 'right')     

    for idx,sim in enumerate(simulations):
        seg = Segment(x0="px", y0="py", x1="px1", y1="py1",line_color={'field': color_field, 'transform': color_mapper},
                name=f'{sim.stats["time"]:0.0f}',tags=[idx],line_width=2 )
        Ptrajectory.add_glyph(s_data[idx],seg,selection_glyph=seg,nonselection_glyph=seg,name=str(idx))

        for i in subplots:
            lin = Line(x='time',y=i.name,line_alpha=.1)
            i.add_glyph(s_data[idx],lin,selection_glyph=lin,nonselection_glyph=lin,name=str(idx))
        

        #if idx>2:
        #    pass
        #    break
    Ptrajectory.circle(x=0, y=0, radius=sim.parameter['target_size'], line_color="#3288bd", fill_color="black", line_width=0,alpha=0.5)

    color_bar = ColorBar(color_mapper=color_mapper,
                        location=(0,1))
    Ptrajectory.add_layout(color_bar, 'right')   
            
    Pentropyp = ColumnDataSource(data=dict(x=[], y=[]))
    Pentropy.circle('x', 'y', source=Pentropyp, alpha=1,size=7)
    PentropyRp = ColumnDataSource(data=dict(x=[], y=[]))
    PentropyR.circle('x', 'y', source=PentropyRp, alpha=1,size=7)
    PentropyPhip = ColumnDataSource(data=dict(x=[], y=[]))
    PentropyPhi.circle('x', 'y', source=PentropyPhip, alpha=1,size=7)

    Ptrajectoryp = ColumnDataSource(data=dict(x=[], y=[]))
    Ptrajectory.circle('x', 'y', source=Ptrajectoryp, alpha=1,size=7)

    Plikelihoodp = ColumnDataSource(data=dict(x=[], y=[]))
    Plikelihood.circle('x', 'y', source=Plikelihoodp, alpha=1,size=7,color='black')

    Pspecialp = ColumnDataSource(data=dict(x=[], a=[],c=[],b=[]))
    Pspecial.line('x', 'a', source=Pspecialp,color='black',legend_label='Total')
    Pspecial.line('x', 'b', source=Pspecialp,color='blue',legend_label='R')
    Pspecial.line('x', 'c', source=Pspecialp,color='green',legend_label='Phi')

    GLOBAL = {'last_idx':-1,'reset':1}
    callback = CustomJS(args={'GLOBAL':GLOBAL,'Ptrajectory':Ptrajectory,'subplots':subplots,'Pspecial':Pspecial,'Plikelihood':Plikelihood,'Rframe':Rframe,'s_likelihood':s_likelihood}, code=code)
    callbackh = CustomJS(args={'Pspecial':Pspecial}, code=codeh)

    taptool = TapTool(callback=callback)
    Ptrajectory.add_tools(taptool)
    for i in subplots:
        taptool = TapTool(callback=callback)
        #HoverTool(tooltips=None,callback=callbackh)
        i.add_tools(taptool)
    ent = column(subplots)
    return row(column(Ptrajectory,Pspecial),Plikelihood,ent)







def plot_multiindex(data,plot_ind,namex=None,namey=None,namez=None,nameC=None,share=False,**args):
    if namex is not None:
        vX = np.unique(data.index.get_level_values(namex))
    else:
        vX = [0]
    if namey is not None:
        vY = np.unique(data.index.get_level_values(namey))
    else:
        vY = [0]
    if namez is not None:
        vZ = np.unique(data.index.get_level_values(namez))
        #valZ = vZ[numz]
    else:
        vZ = [0]
        #valZ = 0
    if nameC is not None:
        tmppara = data.iloc[0]['parameter']
        tmpstats = data.iloc[0]['stats']
        if nameC in data.index.names:
            val_to_color = {val:colors[n%len_colors] for n,val in 
                            enumerate(np.unique(data.index.get_level_values(nameC)))}
            color_type = 'index'
        elif nameC in list(tmppara[0].keys()):
            val_to_color = {val:colors[n%len_colors] for n,val in 
                            enumerate(np.unique(  [p[nameC] for idx,row in data.iterrows() for p in row['parameter']]  ))}
            color_type = 'para'
        elif nameC in list(tmpstats[0].keys()):
            val_to_color = {val:colors[n%len_colors] for n,val in 
                            enumerate(np.unique(   [p[nameC] for idx,row in data.iterrows() for p in row['stats']] ))}
            color_type = 'stats'

    
    fitres = []
    fitres_idx = []
    

    master = True
    ALL_FIGS = []
    for iy in range(len(vY)):
        FIGS_Y = []
        for ix in range(len(vX)):
            if share:
                if master:
                    master = False
                    ax = plot_ind.create()
                    axMaster = ax
                else:
                    ax = plot_ind.create(axMaster)
            else:
                ax = plot_ind.create()
                
            


            if namey is not None and namex is not None:
                idx = {namex:vX[ix],namey:vY[iy]}
                ax.add_layout(Title(text=f"{namey} {vY[iy]}", align="center"), "left")
                ax.add_layout(Title(text=f"{namex} {vX[ix]}", align="center"), "above")
            elif namex is not None:
                idx = {namex:vX[ix]}
                ax.add_layout(Title(text=f"{namex} {vX[ix]}", align="center"), "above")
            else:
                idx = {}
            try:
                if set(idx.keys()) == set(data.index.names):
                    sel_stats_all = data.loc[[tuple((idx[name] if name in idx.keys() else slice(None) for name in data.index.names))]]
                else:
                    sel_stats_all = data.loc[tuple((idx[name] if name in idx.keys() else slice(None) for name in data.index.names)),:]
                    sel_stats_all.index = sel_stats_all.index.droplevel(list(idx.keys()))
            except KeyError as e:
                print(idx,"not found",e)
                continue
                
            try:
                tmpindex = sel_stats_all.index.droplevel(namez)
                try:
                    tmpindex.levshape
                except:
                    tmpindex = pd.Index(np.unique(tmpindex),name=tmpindex.name)
            except:
                tmpindex = [None]
    
            

            for i_idx_z,idx_tmp in enumerate(tmpindex):
                if idx_tmp is None:
                    sel_stats = sel_stats_all
                    label = None
                    idx_z = {}
                else:
                    if not isinstance(idx_tmp, (collections.Sequence, np.ndarray)):
                        idx_tmp= [idx_tmp]
                    idx_z = {tmpindex.names[i_n]:i for i_n,i in enumerate(idx_tmp)}
                    sel_stats = sel_stats_all.loc[tuple((idx_z[name] if name in idx_z.keys() else slice(None) for name in sel_stats_all.index.names)),:]
                    sel_stats.index = sel_stats.index.droplevel(list(idx_z.keys()))
                    label = ' '.join([f'{tmpindex.names[n]}: {i}' for n,i in enumerate(idx_tmp)])
                
                if plot_ind.individual:
                    for idx_l,row in sel_stats.iterrows():
                        if isinstance(sel_stats.index,pd.MultiIndex):
                            idx_all = {sel_stats.index.names[i]:val for i,val in enumerate(idx_l)}
                        else:
                            idx_all = {sel_stats.index.name:idx_l}
                        idx_all.update(idx)
                        idx_all.update(idx_z)

                        if nameC is not None:
                            if color_type == 'para':
                                color = [val_to_color[p[nameC]  ] for p in row['parameter']]
                            elif color_type == 'index':
                                color = val_to_color[idx_all[nameC]]
                            elif color_type == 'stats':
                                color = [val_to_color[p[nameC]  ] for p in row['stats']]
                        else:
                            color = [colors[i%len_colors] for i in range(len(row['parameter']))]
                        args['color'] = color
                        plot_ind.plot(ax,idx_all,row,**args)
                else:
                    if nameC is not None:
                        nametonum = {i:i_n for i_n,i in enumerate(sel_stats.index.names)}
                        color = [val_to_color[idx[nametonum[nameC]] ] for idx in sel_stats.index ]
                    else:
                        color =  None
                    args['colorm'] = color
                    plot_ind.plot(ax,sel_stats,**args)
            FIGS_Y.append(ax)
            #if nameC is not None:     
            #    lines = [Line2D([0], [0], label=f'{nameC} {n:.2e}', color=v) for n,v in val_to_color.items()]
            #    ax.legend(handles=lines)
        ALL_FIGS.append(FIGS_Y)
    return ALL_FIGS




def extract_key(args,key):
    try:
        val =  args[key]
        del args[key]
        return val
    except KeyError:
        return False
    

class plot_lab():
    individual = True
    def __init__(self,maxT=None) :
        self.maxT = maxT

    def create(self):
        f =  figure(tools = 'pan, wheel_zoom, reset',match_aspect = True ,aspect_scale=1)
        f.add_tools(BoxZoomTool(match_aspect=True))
        
        return f

    def plot(self,ax,idx_all,row,**argsi):
        Drot = extract_key(argsi,'Drot')
        stat = extract_key(argsi,'stats')
        maxLines = extract_key(argsi,'maxLines')
        increase = max(int(len(row['parameter'])/maxLines),1)
        
        if self.maxT is None:
            maxTidx = 99999999
        else:
            maxTidx = np.where(row['time']>self.maxT)[0]
            if len(maxTidx)==0:
                maxTidx = 99999999
            else:
                maxTidx = maxTidx[0]
        
        i_n = 0
        while i_n<len(row['parameter']):
            p = row['parameter'][i_n]
            args = argsi.copy()
            if isinstance(argsi['color'],list):
                args['color'] = argsi['color'][i_n]
            if stat:
                if row['stats'][i_n][stat[0]]==stat[1]:
                    continue
            if Drot:
                res = utils.calc_lab_Drot(row['source'][i_n],row['predict'][i_n])
            else:
                res= -row['source'][i_n]
                
            x,y = res.T
            ax.line(x[:maxTidx],y[:maxTidx],**args)
            i_n += increase
        ax.circle(x=[0], y=[0], radius=1, line_color="black", fill_color="white", fill_alpha=0, line_width=2)    
        ax.circle(x=[0], y=[0], radius=p['target_size'], line_color="black", fill_color="orange", fill_alpha=1, line_width=1)
        ax.circle(x=[-p['source'][0]], y=[-p['source'][1]], radius=p['agent_size'], line_color="black", fill_color="green", fill_alpha=.5, line_width=1)
        l = p['l']
        ax.aspect_scale=1
        ax.match_aspect = True 
        #ax.x_range = Range1d(-l,l)
        #ax.y_range = Range1d(-l,l)
        
        

class plot_colum():
    individual = False
    def __init__(self,colum_name,xaxis_name,error=True,tooltips=None,xscale='linear',yscale='linear',legend_loc="top_left",Tbal=True):
        if tooltips is None:
            tooltips = [
            ("speed", "@speed"),
            ("lam", "@lambda_"),
            ("d", "@d"),
            ("agent_size", "@agent_size"),
            ]
        self.tooltips = tooltips
        self.xscale = xscale
        self.yscale = yscale
        self.colum_name = colum_name
        self.error = error
        self.xaxis_name = xaxis_name
        self.legend_loc =  legend_loc
        self.Tbal = Tbal

    def create(self):
        return figure(y_axis_type=self.yscale,x_axis_type=self.xscale,tooltips=self.tooltips)

    def plot(self,ax,data,**argsi):
        loop_idx = data.index.droplevel(self.xaxis_name).drop_duplicates()
        idx_name_to_num ={name:i_n for i_n,name in enumerate(loop_idx.names)}
        Xmin =  99999
        Xmax = -99999
        Ymin =  99999
        Ymax = -99999
        lines = []
        errors = []
        legend_labels = []
        for idx_n,idx in enumerate(loop_idx):
            if isinstance(loop_idx,pd.MultiIndex):
                tmp_idx = tuple([slice(None) if name==self.xaxis_name else idx[idx_name_to_num[name]] for name in data.index.names])
                legend_label = f"".join([f'{name}:{val} ' for val,name in zip(idx,loop_idx.names)])
            else:
                tmp_idx = tuple([slice(None) if name==self.xaxis_name else idx for name in data.index.names])
                legend_label = f"{loop_idx.name}:{idx}"
            legend_labels.append(legend_label)
            sel_data = data.loc[tmp_idx,:]
            X = np.unique(sel_data.index.get_level_values(self.xaxis_name))
            speed = np.array([row[0]['speed']  for _,row in sel_data['parameter'].iteritems()])
            lam = np.array([row[0]['lambda_']  for _,row in sel_data['parameter'].iteritems()])
            d = np.array([row[0]['source']  for _,row in sel_data['parameter'].iteritems()])
            agent_size = np.array([row[0]['agent_size']  for _,row in sel_data['parameter'].iteritems()])
            yn = sel_data[self.colum_name].values
            if self.Tbal:
                dist = np.sqrt(d[:,1]**2+d[:,0]**2) - np.array([row[0]['target_size']  for _,row in sel_data['parameter'].iteritems()]) 
                fdata = pd.DataFrame({'x':X,'y':yn/dist*speed,'agent_size':agent_size,'speed':speed,'lambda_':lam,'d':dist})
            else:
                fdata = pd.DataFrame({'x':X,'y':yn,'agent_size':agent_size,'speed':speed,'lambda_':lam,'d':np.sqrt(d[:,1]**2+d[:,0]**2)})
            if self.error:
                eu =  sel_data[self.colum_name+'_u'].values
                el =  sel_data[self.colum_name+'_l'].values
                fdata['eu'] = eu
                fdata['el'] = el
            source = ColumnDataSource(data=fdata)
            if argsi['colorm'] is None:
                color=colors[idx_n%len_colors]
            else:
                color=argsi['colorm']
            r=ax.line(x="x", y="y",line_width=4,source=source,color=color)
            lines.append(r)
            if self.error:
                eband = Band(base='x', lower='el', upper='eu', source=source, 
                    level='underlay', fill_color=color,fill_alpha=0.5, line_width=1, line_color='black')
                errors.append(eband)
                ax.add_layout(eband)
            else:
                errors.append(None)
            Xmin = min(np.nanmin(fdata['x']),Xmin)
            Xmax = max(np.nanmax(fdata['x']),Xmax)
            Ymin = min(np.nanmin(fdata['y']),Ymin)
            Ymax = max(np.nanmax(fdata['y']),Ymax)
        ax.x_range = Range1d(Xmin*0.9, Xmax*1.1)
        ax.y_range = Range1d(Ymin*0.9, Ymax*1.1)

        ax.xaxis.axis_label = self.xaxis_name
        if self.Tbal:
            ax.yaxis.axis_label = self.colum_name + ' / t_ballistic'
        else:
            ax.yaxis.axis_label = self.colum_name
        #ax.legend.location = self.legend_loc
        #ax.legend.click_policy="hide"
        # Create a legend item for each renderer, and attach a callback to the renderer in order to filter it in/out of view when the legend item is clicked.
        legend_items = []
        for l,r,e in zip(legend_labels,lines,errors):
            if self.error:
                cb = CustomJS(
                args={'render':e},
                code="""
                //console.log(cb_obj);
                var visible = cb_obj.visible;
                // The renderer is currently visibile, but it's legend item was clicked, so the user has requested to make it invisible.
                if (visible) {
                    render.fill_alpha = 0.5;
                    render.line_alpha = 1

                }
                // The renderer is currently invisibile, but it's legend item was clicked, so the user has requested to make it visible.
                else {
                    render.fill_alpha = 0;
                    render.line_alpha = 0
                    //render[0].glyph.line_alpha = 0;
                }
                render.source.change.emit();
                """
                )
                r.js_on_change('visible', cb)
            legend_item = LegendItem(label=l, renderers=[r])
            legend_items.append(legend_item)

        # Create the legend.
        legend = Legend(
            items=legend_items,
            location=self.legend_loc,
            click_policy='hide',
        )

        ax.add_layout(legend)







from . import functions
import re
from bokeh.transform import factor_cmap, factor_mark
class plot_scatter():
    individual = False
    def __init__(self,colum_name,xaxis_exp,yaxis_exp,title='',tooltips=None,xscale='linear',yscale='linear',cscale='linear',
    legend_loc="top_left",Tbal=False,isoSNR=False,reverse=False,minval=None,maxval=None,marker="Drot"):
        if tooltips is None:
            tooltips = [
            (colum_name, "@color"),
            ("lam", "@lam"),
            ("agent_size", "@agent_size"),
            ("Drot", "@Drot"),
            ("SNR tc", "@SNRtc"),
            ("SNR sc", "@SNRsc"),
            ]
        self.tooltips = tooltips
        self.xscale = xscale
        self.yscale = yscale
        self.colum_name = colum_name
        
        self.xaxis_exp = xaxis_exp
        self.yaxis_exp = yaxis_exp
        self.legend_loc =  legend_loc
        self.Tbal = Tbal
        self.title = title 
        self.title += '/ Tballisitc' if Tbal else ''
        self.marker = marker
        self.reverse = reverse
        self.minval = minval
        self.maxval= maxval
        if cscale=='linear':
            self.ColorMapper = LinearColorMapper  
        elif cscale=='log': 
            self.ColorMapper = LogColorMapper
        else:
            print('not supported')
        print(self.title)

    def create(self):
        return figure(title=self.title,y_axis_type=self.yscale,x_axis_type=self.xscale)

    def plot(self,ax,data,**argsi):
        Xmin =  99999
        Xmax = -99999
        Ymin =  99999
        Ymax = -99999
        lines = []
        errors = []
        legend_labels = []
        datascatter = []
        
        for idx,row in data.iterrows():
            para = row['parameter'][0].copy()
            try:
                para['lam'] = para['lambda_']
                del para['lambda_']
            except KeyError:
                pass
            d =  para['source']
            para['sx'] = d[0]
            para['sy'] = d[1]
            para['dist'] = np.sqrt(d[1]**2+d[0]**2)   
            para['gradc'] = (functions.grad_field(1,np.array([d[0],d[1]]))**2).sum() 
            para['c'] =   functions.field(1, para['dist']) 
            para['SNRtc'] = utils.calc_snr(1/para['Drot'],para['lam'],para['speed'],d[0],d[1])
            para['SNRsc'] = utils.calc_snr_sc(1/para['Drot'],para['lam'],para['agent_size'],d[0],d[1])
            if self.Tbal:
                para['color'] = row[self.colum_name]/(para['dist']- para['target_size'])*para['speed'] #The ballistic travel distance is measured till target
            else:
                para['color'] = row[self.colum_name]
            datascatter.append( para )
           
        s_data = pd.DataFrame(datascatter)
        
        l = list(s_data.columns)
        l.sort(key=lambda x:len(x),reverse=True)
        prog  = re.compile(f"({'|'.join(l)})")
        s_data['x'] = eval( re.sub(prog,'s_data["\\1"]',self.xaxis_exp) )
        s_data['y'] = eval( re.sub(prog,'s_data["\\1"]',self.yaxis_exp))
        
        s_data["marker"] = [f"{i}" for i in s_data[self.marker] ]
        LAMS  = np.unique(s_data["marker"])
        MARKERS = ['circle','triangle','hex', 'circle_x']
        
        

        
        #ploting grid lines not working well
        Nextend = 20
        gstep = 1

        for ttt in ['lam','agent_size']:
            i_data = pd.DataFrame()
            i_data[ttt] = 10**np.arange(np.log10(s_data[ttt].min())-Nextend,np.log10(s_data[ttt].max())+Nextend,gstep)
            for col in s_data.columns:
                if col == ttt:
                    continue
                if np.issubdtype(s_data[col].dtype, np.number):
                    i_data[col] = s_data[col].min()    
            for Drot in 10**np.arange(np.log10(s_data['Drot'].min())-Nextend,np.log10(s_data['Drot'].max())+Nextend,gstep):
                i_data['Drot'] = Drot
                i_data['SNRtc'] = [utils.calc_snr(1/row['Drot'],row['lam'],row['speed'],row['sx'],row['sy']) for idx,row in i_data.iterrows()]
                i_data['SNRsc'] =  [utils.calc_snr_sc(1/row['Drot'],row['lam'],row['agent_size']   ,row['sx'],row['sy']) for idx,row in i_data.iterrows()] 
                i_data['x'] = eval( re.sub(prog,'i_data["\\1"]',self.xaxis_exp) )
                i_data['y'] = eval( re.sub(prog,'i_data["\\1"]',self.yaxis_exp))
                gridline = ax.line(x=i_data['x'], y=i_data['y'],color='gray',line_dash='solid')
                gridline.level = 'glyph'



        for ttt in ['lam','Drot']:
            i_data = pd.DataFrame()
            i_data[ttt] = 10**np.arange(np.log10(s_data[ttt].min())-Nextend,np.log10(s_data[ttt].max())+Nextend,gstep)
            for col in s_data.columns:
                if col == ttt:
                    continue
                if np.issubdtype(s_data[col].dtype, np.number):
                    i_data[col] = s_data[col].min()    
            for agent_size in 10**np.arange(np.log10(s_data['agent_size'].min())-Nextend,np.log10(s_data['agent_size'].max())+Nextend,gstep):
                i_data['agent_size'] = agent_size
                i_data['SNRtc'] = [utils.calc_snr(1/row['Drot'],row['lam'],row['speed'],row['sx'],row['sy']) for idx,row in i_data.iterrows()]
                i_data['SNRsc'] =  [utils.calc_snr_sc(1/row['Drot'],row['lam'],row['agent_size']   ,row['sx'],row['sy']) for idx,row in i_data.iterrows()] 
                i_data['x'] = eval( re.sub(prog,'i_data["\\1"]',self.xaxis_exp) )
                i_data['y'] = eval( re.sub(prog,'i_data["\\1"]',self.yaxis_exp))
                gridline = ax.line(x=i_data['x'], y=i_data['y'],color='gray',line_dash='solid')
                gridline.level = 'glyph'


        for ttt in ['agent_size','Drot']:
            i_data = pd.DataFrame()
            i_data[ttt] = 10**np.arange(np.log10(s_data[ttt].min())-Nextend,np.log10(s_data[ttt].max())+Nextend,gstep)
            for col in s_data.columns:
                if col == ttt:
                    continue
                if np.issubdtype(s_data[col].dtype, np.number):
                    i_data[col] = s_data[col].min()    
            for lam in 10**np.arange(np.log10(s_data['lam'].min())-Nextend,np.log10(s_data['lam'].max())+Nextend,gstep):
                i_data['lam'] = lam
                i_data['SNRtc'] = [utils.calc_snr(1/row['Drot'],row['lam'],row['speed'],row['sx'],row['sy']) for idx,row in i_data.iterrows()]
                i_data['SNRsc'] =  [utils.calc_snr_sc(1/row['Drot'],row['lam'],row['agent_size']   ,row['sx'],row['sy']) for idx,row in i_data.iterrows()] 
                i_data['x'] = eval( re.sub(prog,'i_data["\\1"]',self.xaxis_exp) )
                i_data['y'] = eval( re.sub(prog,'i_data["\\1"]',self.yaxis_exp))
                gridline = ax.line(x=i_data['x'], y=i_data['y'],color='gray',line_dash='solid')
                gridline.level = 'glyph'


        #end plotting grid
        
        
        source = ColumnDataSource(data=s_data)
        
        minval = min(s_data['color']) if self.minval is None else self.minval
        maxval = max(s_data['color']) if self.maxval is None else self.maxval

        if not self.reverse:
            color_mapper = self.ColorMapper(palette='Viridis256', low=minval, high=maxval)
        else:
            paltet = bokeh.palettes.Viridis256[::-1]
            color_mapper = self.ColorMapper(palette=paltet, low=minval, high=maxval)
        s1 = ax.scatter(x="x", y="y",source=source,
            color={'field': 'color', 'transform': color_mapper},
            fill_alpha=0.9, size=10,
            marker=factor_mark('marker', MARKERS[:len(LAMS)], LAMS),legend_group="marker" )
        
        color_bar = ColorBar(color_mapper=color_mapper, ticker= BasicTicker(),
                     location=(0,0))
        ax.add_layout(color_bar, 'right')
        ax.x_range = Range1d(s_data['x'].min()*0.8, s_data['x'].max()*1.2)
        ax.y_range = Range1d(s_data['y'].min()*0.8, s_data['y'].max()*1.2)
        ax.xaxis.axis_label = self.xaxis_exp
        ax.yaxis.axis_label = self.yaxis_exp
        ax.legend.title=self.marker
        ax.legend.location = self.legend_loc
        
        hover = HoverTool(renderers=[s1], tooltips=self.tooltips, mode='mouse') 
        ax.add_tools(hover)
        
class plot_colum_t():
    individual = False
    def __init__(self,colum_name,colum_idx=None,tooltips=None,ploti=False,typ='dashed',xscale='linear',yscale='linear',legend_loc="top_left",Xsnr=False,Yoffset=0):
        if tooltips is None:
            tooltips = [
            ("", "$name"),
            ]
        self.tooltips = tooltips
        self.xscale = xscale
        self.yscale = yscale
        self.colum_name = colum_name
        self.colum_idx = colum_idx
        self.typ = typ
        self.legend_loc =  legend_loc
        self.ploti = ploti
        self.Xsnr = Xsnr
        self.Yoffset = Yoffset

    def create(self):
        return figure(y_axis_type=self.yscale,x_axis_type=self.xscale,tooltips=self.tooltips)

    def plot(self,ax,data,**argsi):
        loop_idx = data.index
        
        Xmin =  99999
        Xmax = -99999
        Ymin =  99999
        Ymax = -99999
        lines = []
        linesi = []
        legend_labels = []

        if argsi['colorm'] is None:
            color=[colors[i%len_colors] for i in range(len(loop_idx))]
        else:
            color=argsi['colorm']

        for idx_n,idx in enumerate(loop_idx):
            if isinstance(loop_idx,pd.MultiIndex):
                legend_label = f"".join([f'{name}:{val} ' for val,name in zip(idx,loop_idx.names)])
            else:
                legend_label = f"{loop_idx.name}:{idx}"
            
            legend_labels.append(legend_label)
            sel_data = data.loc[idx]
            Y = sel_data[self.colum_name].values - self.Yoffset
            
            if self.Xsnr:
                X = np.sqrt(utils.calc_snr(np.array(sel_data['time'][:len(Y)]),1,0.01,0,0.2)*np.exp(-2.03935462)**2)
            else:
                X = sel_data['time'][:len(Y)]
            speed = sel_data['parameter'][0]['speed']  
            lam = sel_data['parameter'][0]['lambda_'] 
            d = sel_data['parameter'][0]['source']  
            agent_size = sel_data['parameter'][0]['agent_size'] 
            Drot = sel_data['parameter'][0]['Drot'] 
            name = f'Drot {Drot},agent_size {agent_size}'
           
            if self.ploti:
                tmplines = []
                for i in sel_data['source']:
                    d = np.sqrt(i[:,0]**2+i[:,1]**2)
                    patch = ax.line( sel_data['time'][:len(d)], d,line_width=1,alpha=0.5,color=color[idx_n])
                    #patch.level = 'underlay'
                    tmplines.append(patch)
                linesi.append(tmplines)
            else:
                linesi.append([])
            if self.colum_idx is None:
                source = ColumnDataSource(data={'X':X,'Y':Y})
                r=ax.line(x="X", y="Y",line_width=2,alpha=1,source=source,color=color[idx_n],name=name)
                r.level ='guide'
                lines.append([r])
            else:
                c_idx =  sel_data[self.colum_idx]
                source = ColumnDataSource(data={'X':X[:c_idx],'Y':Y[:c_idx]})
                r=ax.line(x="X", y="Y",line_width=2,alpha=1,source=source,color=color[idx_n],name=name)
                r.level ='guide'
                if self.typ is None:
                    Xmin = min(np.nanmin(X[:c_idx]),Xmin)
                    Xmax = max(np.nanmax(X[:c_idx]),Xmax)
                    Ymin = min(np.nanmin(Y[:c_idx]),Ymin)
                    Ymax = max(np.nanmax(Y[:c_idx]),Ymax)
                    lines.append([r])
                else:
                    sourced = ColumnDataSource(data={'X':X[c_idx:],'Y':Y[c_idx:]})
                    rd=ax.line(x="X", y="Y",line_width=1,alpha=1,line_dash=self.typ,source=sourced,color=color[idx_n],name=name)
                    #rd.level ='guide'
                    lines.append([r,rd])
                    Xmin = min(np.nanmin(X),Xmin)
                    Xmax = max(np.nanmax(X),Xmax)
                    Ymin = min(np.nanmin(Y),Ymin)
                    Ymax = max(np.nanmax(Y),Ymax)
        ax.x_range = Range1d(Xmin*0.9, Xmax*1.1)
        #ax.x_range = Range1d(0, 400)
        ax.y_range = Range1d(Ymin*0.9, Ymax*1.1)
        ax.xaxis.axis_label = 'time'
        ax.yaxis.axis_label = self.colum_name
        #ax.legend.location = self.legend_loc
        #ax.legend.click_policy="hide"
        # Create a legend item for each renderer, and attach a callback to the renderer in order to filter it in/out of view when the legend item is clicked.
        legend_items = []
        for l,r,ri in zip(legend_labels,lines,linesi):
            legend_item = LegendItem(label=l, renderers=r+ri)
            legend_items.append(legend_item)
        # Create the legend.
        legend = Legend(
            items=legend_items,
            location=self.legend_loc,
            click_policy='hide',
        )

        ax.add_layout(legend)



class plot_entropy():
    individual = True
    def __init__(self) -> None:
        pass

    def create(self):
        return figure()

    def plot(ax,idx_all,row,**argsi):
        extend = extract_key(argsi,'extend')
        error = extract_key(argsi,'error')
        fit = extract_key(argsi,'fit')
        typ = 'entropy_phi'
        
        sources = row["source"]
        para = row['parameter']
        peaks = row['idxswitch_c']
        peakmax = int(np.nanmax(peaks))
        var = row[typ]

        all_data = []
        for s_n,p in enumerate(para):
            idx_s = peaks[s_n]
            if np.isfinite(idx_s):
                #ax.plot(sel_stats.loc[speed,"timei"][s_n][:idx_s],
                #        sel_stats.loc[speed,"entropy_phi"][s_n][:idx_s],
                #        color=colors[phis_to_num[p["phi0"]]],lw=1,alpha=0.05,zorder=1)
                all_data.append(pd.Series(var[s_n][:idx_s],index=row["timei"][s_n][:idx_s]))
                #all_data.append(pd.Series(var[s_n][:peakmax],index=sel_stats["timei"][s_n][:peakmax]))
                pass
            else:
                pass
                #ax.plot(xs,ys)
        tmp = pd.concat(all_data,axis=1)
        tmp_ind = np.where(tmp.notna().mean(axis=1)<0.7)[0][0]
        #tmp_ind = len(tmp)
        meand = tmp.mean(axis=1)
        stdd = tmp.std(axis=1)
        Ystdd = stdd.values
        #if typ == 'entropy_phi':
        #    X = utils.calc_snr(np.array(meand.index),p['lambda_'],p['speed'],p['source'][0],p['source'][1])
        #elif typ == 'entropy':
        #    X = np.array(meand.index) * p['lambda_']
        X=meand.index
        Y = np.log2(2*np.pi)-meand.values
        source = ColumnDataSource(data=dict(
        x=X[:tmp_ind],
        y=Y[:tmp_ind]
        ))
        l = ax.line('x','y',source=source,name=''.join([f'{n}: {v}, ' for n,v in idx_all.items()]),**argsi)
        
        if extend:
            ax.plot(X[tmp_ind:],Y[tmp_ind:],color='gray',zorder=4)

        if error:
            ax.fill_between(X[:tmp_ind], (Y-Ystdd)[:tmp_ind] , (Y+Ystdd)[:tmp_ind],
                        alpha=0.2,zorder=1,color=l.get_color())

        if fit:
            model = sm.OLS(Y[:tmp_ind],sm.add_constant(X[:tmp_ind]))
            results = model.fit()
            fitres = results.params[1]
            fitres_idx = [idx_all[k] for k in idx_all.keys()]
            ax.plot(X[:tmp_ind],X[:tmp_ind]*results.params[1]+results.params[0],color=l.get_color(),zorder=2,
                    label=f"$g_0$ {results.params[1]:.2e}, $g_1$ {results.params[0]:.2e}")





class plot_ft():
    individual = False
    def __init__(self,tooltips=None):
        if tooltips is None:
            tooltips = [
            ("speed", "@speed"),
            ("lam", "@lambda_"),
            ("d", "@d"),
            ("agent_size", "@agent_size"),
            ]
        self.tooltips = tooltips
        

    def create(self,):
            
        return figure(y_axis_type="log",x_axis_type="log",tooltips=self.tooltips)

    def plot(self,ax,data,**argsi):
        speed = np.array([row[0]['speed']  for _,row in data['parameter'].iteritems()])
        lam = np.array([row[0]['lambda_']  for _,row in data['parameter'].iteritems()])
        d = np.array([row[0]['source']  for _,row in data['parameter'].iteritems()])
        agent_size = np.array([row[0]['agent_size']  for _,row in data['parameter'].iteritems()])
        yn = data['tswitch_cm'].values
        e =  data['tswitch_cs'].values
        
        fdata = pd.DataFrame({'agent_size':agent_size,'x':lam * speed**2,'y':yn,'e':e,'speed':speed,'lambda_':lam,'dx':d[:,0],'dy':d[:,1],'d':np.sqrt(d[:,1]**2+d[:,0]**2)}).sort_index()
        fdata['t_snr'] = np.array([utils.calc_t_snr(row['lambda_'],row['speed'],row['dx'],row['dy']) for idx,row in fdata.iterrows()] )
        source = ColumnDataSource(data=fdata)
        X = 't_snr'
        ax.scatter(x=X, y="y",source=source)
        ax.y_range = Range1d(fdata['y'].min()*0.9, fdata['y'].max()*1.1)
        ax.x_range = Range1d(fdata[X].min()*0.9, fdata[X].max()*1.1)

        
    

class plot_ftphi0():
    individual = False
    def __init__(self,tooltips=None):
        if tooltips is None:
            tooltips = [
            ("speed", "@speed"),
            ("lam", "@lambda_"),
            ("d", "@d"),
            ("agent_size", "@agent_size"),
            ("phi0", "@phi0")
            ]
        self.tooltips = tooltips
        

    def create(self,):
            
        return figure(y_axis_type="log",x_axis_type="log",tooltips=self.tooltips)

    def plot(self,ax,data,**argsi):
        speed = np.array([row[0]['speed']  for _,row in data['parameter'].iteritems()])
        lam = np.array([row[0]['lambda_']  for _,row in data['parameter'].iteritems()])
        d = np.array([row[0]['source']  for _,row in data['parameter'].iteritems()])
        agent_size = np.array([row[0]['agent_size']  for _,row in data['parameter'].iteritems()])
        phi0 = np.array([row[0]['phi0']  for _,row in data['parameter'].iteritems()])

        yn = data['tswitch_cm'].values
        e =  data['tswitch_cs'].values
        
        fdata = pd.DataFrame({'phi0':phi0,'agent_size':agent_size,'x':lam * speed**2,'y':yn,'e':e,'speed':speed,'lambda_':lam,'dx':d[:,0],'dy':d[:,1],'d':np.sqrt(d[:,1]**2+d[:,0]**2)}).sort_index()
        fdata['t_snr'] = np.array([utils.calc_t_snr(row['lambda_'],row['speed'],row['dx'],row['dy']) for idx,row in fdata.iterrows()] )
        fdata['color'] = argsi['color']
        source = ColumnDataSource(data=fdata)
        X = 't_snr'
        ax.scatter(x=X, y="y",source=source,color='color')
        ax.y_range = Range1d(fdata['y'].min()*0.9, fdata['y'].max()*1.1)
        ax.x_range = Range1d(fdata[X].min()*0.9, fdata[X].max()*1.1)


class plot_hexbin():
    individual = True
    def __init__(self,size=200) :
        self.size = size

    def create(self,master=None):
        if master is None:
            self.master = True
            f =  figure(tools = 'pan, wheel_zoom, reset',match_aspect = True ,aspect_scale=1,plot_width=self.size,plot_height=self.size)
        else:
            self.master = False
            f =  figure(tools = 'pan, wheel_zoom, reset',match_aspect = True ,aspect_scale=1,plot_width=self.size,plot_height=self.size, 
            x_range=master.x_range, y_range=master.y_range)

        f.add_tools(BoxZoomTool(match_aspect=True))
        return f

    def plot(self,ax,idx_all,row,**argsi):
        X=[]
        Y=[]
        for s,a in zip(row['source'],row['predict']):
            x,y = utils.calc_lab_Drot(s,a).T
            X.append(x)
            Y.append(y)
        X=np.hstack(X)
        Y=np.hstack(Y)       
        r, bins = ax.hexbin(X, Y, size=0.02, palette='Turbo256')
        ax.aspect_scale=1
        ax.match_aspect = True 
        l=row['parameter'][0]['l']
        if self.master:
            ax.x_range = Range1d(-l,l)
            ax.y_range = Range1d(-l,l)
        