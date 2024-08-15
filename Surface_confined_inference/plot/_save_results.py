import matplotlib.pyplot as plt
import numpy as np
import Surface_confined_inference as sci
from pandas import DataFrame
def save_results(time, voltage, experiment, simulations, directory, experiment_type,boundaries, **kwargs):
    if "save_csv" not in kwargs:
        kwargs["save_csv"]=False
    if "harmonics" not in kwargs:
        if experiment_type in ["PSV", "FTACV"]:
          kwargs["harmonics"]=list(range(1, 11))
    if "DC_voltage" not in kwargs:
        kwargs["DC_voltage"]=None
    if "table" not in kwargs:
        kwargs["table"]=True
    necessary_args=set(["optim_list", "fixed_parameters", "score", "parameters"])
    kwarg_keys=set(kwargs.keys())
    if kwargs["table"]==True:
        if necessary_args.issubset(kwarg_keys) is False:
            raise ValueError("Need {0} for a table".format(kwarg_keys-necessary_args))
        else:
            labels=["Rounded_table.txt", "Full_table.txt"]
            dp=[2, 10]
            
            for m in range(0, 2):
                table_titles=["Rank"]+kwargs["optim_list"]+list(kwargs["fixed_parameters"].keys())+["Dimensionless score"]
                fancy_titles=sci._utils.get_titles(table_titles, units=True)
                max_len= [len(x) for x in fancy_titles]
                str_values=[]
                
                for i in range(0, len(simulations)):
                    full_list={**dict(zip(kwargs["optim_list"], kwargs["parameters"][i,:])), **kwargs["fixed_parameters"], **{"Dimensionless score":kwargs["score"][i], "Rank":i+1}}
                    formatted_values=[sci._utils.format_values(full_list[key], dp=dp[m]) for key in table_titles]
                    str_values.append(formatted_values)
                    if m==0:
                        if i==0:
                            bar_list={key:[full_list[key]] for key in full_list if key != "Rank" }
                        else:
                            for key in full_list.keys():
                                if key != "Rank":
                                    bar_list[key].append(full_list[key])
                    for j in range(0, len(max_len)):
                        max_len[j]=max(len(formatted_values[j]), max_len[j])
                with open(directory+"/{0}".format(labels[m]), "w") as f:
                    table_writer(fancy_titles, max_len, f)
                    for values in str_values:
                        table_writer(values, max_len, f)
            param_keys=table_titles[1:]
            dimensions=sci._utils.det_subplots(len(param_keys))
            bar_fig, bar_ax=plt.subplots(*dimensions)
            for i in range(0, len(param_keys)):
                axis=bar_ax[i//dimensions[1], i%dimensions[1]]
                axis.scatter(range(1, len(simulations)+1), bar_list[param_keys[i]])
                axis.set_xlabel("Rank")
                ylabel=fancy_titles[i+1]
                if param_keys[i] in kwargs["fixed_parameters"].keys():
                    ylabel+=" (pinned)"
                elif param_keys[i] != "Dimensionless score":
                    axis.set_ylim([boundaries[param_keys[i]][0], boundaries[param_keys[i]][1]])
                axis.set_ylabel(ylabel)
            adjust_and_save(bar_fig, directory, "Parameter scatter plot.png", size=(8, 5), hspace=0.3, wspace=0.7)

               
                
            
    if len(np.array(simulations).shape)==1:
        pooled_figure=False
    else:
        pooled_figure=True
    
    label_dict={"time":"Time (s)", "voltage":"Potential (V)"}
    if experiment_type =="FTACV":
        harm_func=np.abs
        harmonic=True
        hanning=True
        xaxis=time
        xlabel="time"
    elif experiment_type=="PSV":
        harmonic=True
        harm_func=np.real
        xaxis=voltage
        hanning=False
        xlabel="voltage"
    elif experiment_type=="DCV":
        harmonic=False
        xaxis=voltage
        xlabel="voltage"
    
    
    if pooled_figure==True:
        pool_fig, pool_ax=plt.subplots()
        pool_ax.set_xlabel(label_dict[xlabel])
        pool_ax.set_ylabel("Current (A)")
        pool_ax.plot(xaxis, experiment, label="Data")
        if harmonic==True:
            pooled_h_fig, pooled_h_ax=plt.subplots(len(kwargs["harmonics"]), 1)
            pooled_plot_dict=dict(
                Experimental_data={"time": time, "current": experiment, "voltage":voltage, "harmonics":kwargs["harmonics"],},
                xaxis=xlabel,
                hanning=hanning,
                plot_func=harm_func,
                
                xlabel=label_dict[xlabel],
                ylabel="Current (A)",
                axes_list=pooled_h_ax
                )
    for i in range(0, len(simulations)):
        try:
            fig, ax=plt.subplots()
            ax.plot(xaxis, experiment, label="Data")
            ax.plot(xaxis, simulations[i,:], label="Simulation")
            if kwargs["save_csv"]==True:
                save_dict={"Time (s)":time, "Potential (V)":voltage, "Current (A)":simulations[i,:]}
                if kwargs["DC_voltage"] is not None:
                    if experiment_type=="FTACV":
                        save_dict["DC Potential (V)"]=kwargs["DC_voltage"]
            ax.set_xlabel(label_dict[xlabel])
            ax.set_ylabel("Current (A)")
            ax.legend()
            adjust_and_save(fig, directory, "Rank {0} current.png".format(i+1))
            if pooled_figure==True:
                pool_ax.plot(xaxis, simulations[i,:], label="Rank {0}".format(i+1), alpha=0.5)
            if harmonic==True:
                h_fig, h_ax=plt.subplots(len(kwargs["harmonics"]), 1)
                plot_dict=dict(
                    Experimental_data={"time": time, "current": experiment, "voltage":voltage},
                    xaxis=xlabel,
                    hanning=hanning,
                    plot_func=harm_func,
                    harmonics=kwargs["harmonics"],
                    xlabel=label_dict[xlabel],
                    ylabel="Current (A)",
                    axes_list=h_ax
                    )
                plot_dict["Rank {0}_data".format(i+1)]={"time": time, "current": simulations[i,:], "voltage":voltage}
                sci.plot.plot_harmonics(**plot_dict)   
                if pooled_figure==True:
                    pooled_plot_dict["Rank {0}_data".format(i+1)]={"time": time, "current": simulations[i,:], "voltage":voltage}
                adjust_and_save(h_fig, directory,"Rank {0} harmonics.png".format(i+1))
                if kwargs["save_csv"]==True:
                    generated_harmonics=sci.plot.generate_harmonics(time, 
                                                            simulations[i,:],
                                                            hanning=hanning,
                                                            plot_func=harm_func,
                                                            harmonics=kwargs["harmonics"],
                                                            )
                    for j in range(0, len(kwargs["harmonics"])):
                        save_dict["Harmonic {0}".format(kwargs["harmonics"][j])]=harm_func(generated_harmonics[j,:])
            if kwargs["save_csv"]==True:
                DataFrame(save_dict).to_csv(directory+"/"+"Rank {0}.csv".format(i+1))
        except:
            with open(directory+"/"+"failed_runs.txt", "a") as f:
                f.write("{1} Run {0} failed\n".format(i+1, experiment_type))

            continue
    
    if pooled_figure==True:
        pool_ax.legend()
        adjust_and_save(pool_fig, directory, "Pooled current.png")
        if harmonic==True:
        
            sci.plot.plot_harmonics(**pooled_plot_dict)  
            adjust_and_save(pooled_h_fig, directory, "Pooled harmonic current.png")


def adjust_and_save(figure, directory, name, size=(8,8), hspace=0, wspace=0.2):
    adjusted_plots=dict(top=0.969,
                        bottom=0.081,
                        left=0.146,
                        right=0.977,
                        hspace=hspace,
                        wspace=wspace)
    
    figure.set_size_inches(size[0], size[1])
    figure.subplots_adjust(**adjusted_plots)
    figure.savefig(directory+"/"+name, dpi=500)
    figure.clf()
    plt.close(figure)
def table_writer(write_list, max_len, file):
    title_list=["" for x in range(0,len(write_list))]

    for i in range(0, len(write_list)):
        num_space=max_len[i]-len(write_list[i])
        title_list[i]=write_list[i]+","+" "*num_space
    write_string=" ".join(title_list)+"\n"
    file.write(write_string)
    
