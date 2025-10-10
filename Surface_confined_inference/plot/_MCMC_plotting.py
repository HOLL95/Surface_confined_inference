import copy

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pints

import Surface_confined_inference as sci


def change_param( params, optim_list, parameter, value):
    param_list=copy.deepcopy(params)
    param_list[optim_list.index(parameter)]=value
    return param_list
def chain_appender( chains, param, burn=False):
    if burn is not False:
        burnlen=burn
    else:
        burnlen=0
    new_chain=chains[0, burnlen:, param]
    for i in range(1, len(chains)):
        new_chain=np.append(new_chain, chains[i, burnlen:, param])
    return new_chain
def convert_to_zscore( chains, burn=False):
    if burn is False:
        new_chain=copy.deepcopy(chains)
    else:
        new_chain=np.zeros((chains.shape[0], chains.shape[1]-burn, chains.shape[2]))
    for i in range(0, len(chains)):
        for j in range(0, len(chains[i,0,:])):
            current_samples=chains[i, burn:, j]
            mean=np.mean(current_samples)
            std=np.std(current_samples)
            new_chain[i, :, j]=np.subtract(current_samples, mean)/std
    return new_chain
def concatenate_all_chains( chains, burn=0):
    return [sci.plot.chain_appender(chains, x, burn) for x in range(0, len(chains[0, 0, :]))]
def plot_params( titles, set_chain, **kwargs):
    if "positions" not in kwargs:
        kwargs["positions"]=range(0, len(titles))
    if "row" not in kwargs:
        row, col=sci._utils.det_subplots(len(titles))
    else:
        row=kwargs["row"]
        col=kwargs["col"]
    if "label" not in kwargs:
        kwargs["label"]=None
    if "axes" not in kwargs:
        fig, ax=plt.subplots(row, col)
    else:
        ax=kwargs["axes"]
    if "alpha" not in kwargs:
        kwargs["alpha"]=1
    if "pool" not in kwargs:
        kwargs["pool"]=True
    if "burn_remove" not in kwargs:
        kwargs["burn_remove"]=True
    if "Rhat_title" not in kwargs:
        kwargs["Rhat_title"]=False
    if "log" not in kwargs:
        kwargs["log"]=False
    if "true_values" not in kwargs:
        kwargs["true_values"]=False        
    if "title_debug" not in kwargs:
        kwargs["title_debug"]=False
    if kwargs["title_debug"] is False:
        titles=sci._utils.get_titles(titles, units=True, positions=kwargs["positions"])
    for i in range(0, len(titles)):
        if len(ax.shape)!=1:
            axes=ax[i//col, i%col]
        else:
            axes=ax[i]
        plot_chain=sci.plot.chain_appender(set_chain, kwargs["positions"][i], burn=kwargs["burn_remove"])#axes.set_title()
        if kwargs["log"]==True:
            
            plot_chain=np.log10(plot_chain)
        if abs(np.mean(plot_chain))<1e-5:
            axes.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2e'))
        elif abs(np.mean(plot_chain))<0.001:
            axes.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1e'))
        else:
            
            order=np.log10(np.std(np.abs(plot_chain)))

            if order>1:
                axes.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
            else:
                
                order_val=abs(int(np.ceil(np.abs(order))))+1
                axes.xaxis.set_major_formatter(ticker.FormatStrFormatter(f'%.{order_val}f'))
        if kwargs["pool"]==True:
            axes.hist(plot_chain,bins=20, stacked=True, label=kwargs["label"], alpha=kwargs["alpha"])
        elif kwargs["pool"]==False:
            for j in range(0, len(set_chain[:, 0, 0])):
                axes.hist(set_chain[j, kwargs["burn_remove"]:, i],bins=20, stacked=True, label=f"Chain {i+1}", alpha=kwargs["alpha"])
        elif kwargs["pool"]=="pre_pooled":
                axes.hist(set_chain,bins=20, stacked=True, label=kwargs["label"], alpha=kwargs["alpha"])
        if kwargs["Rhat_title"]==True:
            rhat_val=rhat(set_chain[:, kwargs["burn_remove"]:, i])
            axes.set_title(round(rhat_val, 3))
        if kwargs["true_values"]!=False:
            axes.axvline(kwargs["true_values"][kwargs["positions"][i]], linestyle="--", color="black")
        #axes.legend()
        lb, ub = axes.get_xlim( )
        axes.set_xticks(np.linspace(lb, ub, 3))
        axes.set_xlabel(titles[i])
        axes.set_ylabel('frequency')
        
        #axes.set_title(graph_titles[i])
    return  ax
def axes_legend( label_list,  ax,**kwargs):
    if "colours" not in kwargs:
        kwargs["colours"]=[None for x in range(0, len(label_list))]
        
    for i in range(0, len(label_list)):
        ax.plot(0, 0, label=label_list[i], color=kwargs["colours"][i])
    if "bbox" not in kwargs:
        ax.legend()
    else:
        ax.legend(bbox_to_anchor=kwargs["bbox"])
    ax.set_axis_off()
def trace_plots( params, chains,**kwargs):
    if "rhat" not in kwargs:
        rhat =False
    else:
        rhat=kwargs["rhat"]
    if "burn" not in kwargs:
        burn=0
    else:
        burn=kwargs["burn"]
    if "order" not in kwargs:
        order=list(range(0, len(params)))
    else:
        
        order=kwargs["order"]
        print(order)
    if "true_values" not in kwargs:
        kwargs["true_values"]=None
    row, col=sci._utils.det_subplots(len(params))
    
    if rhat==True:
        rhat_vals=pints.rhat(chains[:, burn:, :], warm_up=0.5)
    names=sci._utils.get_titles(params, units=False)
    y_labels=sci._utils.get_titles(params, units=True)
    for i in range(0, len(params)):
        axes=plt.subplot(row,col, i+1)
        for j in range(0, len(chains)):
            axes.plot(chains[j, :, order[i]], label="Chain "+str(j), alpha=0.7)
        if abs(np.mean(chains[j, :, order[i]]))<0.01:
            axes.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1e'))
        if "omega" in params[i]:
            axes.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.5f'))
        #else:
        #    axes.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        #axes.set_xticks([0, 10000])
        if kwargs["true_values"] is not None:
            if params[i] in kwargs["true_values"]:
                print(params[i], params[i])
                axes.axhline(kwargs["true_values"][params[i]], color="black", linestyle="--")
        if i>(len(params))-(col+1):
            axes.set_xlabel('Iteration')
        #if i==len(titles)-1:
        #    axes.legend(loc="center", bbox_to_anchor=(2.0, 0.5))
        #lb, ub = axes.get_xlim( )
        #axes.set_xticks(np.linspace(lb, ub, 5))
        if rhat == True:
            axes.set_title("$\\hat{R}$="+str(round(rhat_vals[i],2))+"")
        else:
            axes.set_title(names[i])
        
        axes.set_ylabel(y_labels[i])
def plot_2d( params, chains, **kwargs):
    n_param=len(params)
    if "pooling" not in kwargs:
        kwargs["pooling"]=False
    if "burn" not in kwargs:
        burn=0
    else:
        burn=kwargs["burn"]
    if "order" not in kwargs:
        kwargs["order"]=list(range(0, n_param))
    if "true_values" not in kwargs:
        kwargs["true_values"]=None
    if "density" not in kwargs:
        kwargs["density"]=False
    if "nbins" not in kwargs:
        kwargs["nbins"]=None
    if "log" not in kwargs:
        kwargs["log"]=False
    if "twinx" not in kwargs:
        kwargs["twinx"]=None
    
    if "axes" not in kwargs:
        fig, ax=plt.subplots(n_param, n_param)
    else:
        ax=kwargs["axes"]
    if "alpha" not in kwargs:
        kwargs["alpha"]=1
    if "colour" not in kwargs:
        kwargs["colour"]=None
    elif len(kwargs["axis"])!=n_param:
        return ValueError(f"Axis length must be {n_param}")
    else:
        ax=kwargs["axis"]   
    if "title_debug" not in kwargs:
        kwargs["title_debug"]=False

    if kwargs["title_debug"] is False:
        labels=sci._utils.get_titles(params, units=True)
    else:
        labels=params


    chain_results=chains
    
    twinx=[]
    for i in range(0,n_param):
        
        z=kwargs["order"][i]
            
        pooled_chain_i=[chain_results[x, burn:, z] for x in range(0, 3)]
        if kwargs["pooling"]==True: 
            
            chain_i=[np.concatenate(pooled_chain_i)]
        else:
            chain_i=pooled_chain_i
        if "rotation" not in kwargs:
            kwargs["rotation"]=False
        #for m in range(0, len(labels)):
        #    box_params[chain_order[i]][labels[m]][exp_counter]=func_dict[labels[m]]["function"](chain_i, *func_dict[labels[m]]["args"])
        #chain_i=np.multiply(chain_i, values[i])
        
        for j in range(0, n_param):
            m=kwargs["order"][j]
            if i==j:
                axes=ax[i,j]
                axes.set_yticks([])
                if kwargs["twinx"] is None:
                    ax1=axes.twinx()
                else:
                    ax1=kwargs["twinx"][j]
                for z in range(0, len(chain_i)):
                    ax1.hist(chain_i[z], density=kwargs["density"], bins=kwargs["nbins"], log=kwargs["log"],  color=kwargs["colour"])
                twinx.append(ax1)
                #ticks=axes.get_yticks()
                #axes.set_yticks([])
                #ax1.set_yticks(ticks)
                #if kwargs["log"]==True:
                #    ax1.set_yscale("log")
                if kwargs["density"] is False:
                    ax1.set_ylabel("Frequency")
                else:
                    ax1.set_ylabel("Density")
            elif i<j:
                ax[i,j].axis('off')
            else:
                axes=ax[i,j]
                if kwargs["log"] is True:
                    axes.set_yscale('log')
                    axes.set_xscale('log')
                chain_j=[chain_results[x, burn:, m] for x in range(0, 3)]
                if kwargs["pooling"]==True: 
                    chain_j=[np.concatenate(chain_j)]
                for z in range(0, len(chain_i)):
                    axes.scatter(chain_j[z], chain_i[z], s=0.5, alpha=kwargs["alpha"], color=kwargs["colour"])
                    #print(axes.get_xlim(), labels[i], labels[j])
            if kwargs["true_values"] is not None:
                
                if params[j] in kwargs["true_values"] and params[i] in kwargs["true_values"]:
                    if i>j:
                        axes.scatter(kwargs["true_values"][params[j]],kwargs["true_values"][params[i]], color="black", s=20, marker="x")
                    elif i==j:
                        axes.axvline(kwargs["true_values"][params[i]], color="black", linestyle="--")

            """ if i!=0:
                if chain_order[i]=="CdlE3":
                    ax[i, 0].set_ylabel(titles[i], labelpad=20)
                elif chain_order[i]=="gamma":
                    ax[i, 0].set_ylabel(titles[i], labelpad=30)
                else:
                    """
            if i<n_param-1:
                ax[i,j].set_xticklabels([])#
            if j>0 and i!=j:
                ax[i,j].set_yticklabels([])
            if j!=n_param:
                ax[-1, i].set_xlabel(labels[i])
                if np.mean(np.abs(chain_i))<1e-4:
                    ax[-1, i].xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1e'))
                if kwargs["rotation"] is not False:
                    plt.setp( ax[-1, i].xaxis.get_majorticklabels(), rotation=kwargs["rotation"] )
            if i!=0:
                ax[i, 0].set_ylabel(labels[i])
                if np.mean(np.abs(chain_i))<1e-4:
                    ax[i, 0].yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1e'))
                
    #print("++", len(twinx))   

    return ax, twinx
def convert_idata_to_pints_array(idata):
    chains=idata.to_dict()
    params=list(chains["posterior"].keys())
    num_params=len(params)
    num_chains=len(chains["posterior"][params[0]])
    num_samples=len(chains["posterior"][params[0]][0])
    empty_pints=np.zeros((num_chains, num_samples, num_params))

    for i in range(0, num_params):
        key=params[i]
        for j in range(0, num_chains):
            empty_pints[j, :, i]=chains["posterior"][key][j]
    return empty_pints

